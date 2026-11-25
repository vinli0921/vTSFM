"""Resumable eager FP32 Moirai 2 quantile forecasts for independent univariate rows.

Each owner keeps the native true-length batch geometries. The first call
has one trajectory per row; subsequent calls carry nine and reduce all 81
candidates. No attention KV or token-prefix reuse is assumed.
"""
from dataclasses import dataclass
import hashlib
import inspect
import os

import torch
from uni2ts.model.moirai2 import Moirai2Forecast

from .admission import DEFAULT_LIMITS, OUTPUT_BLOCK_SIZE, PATCH_SIZE

# Pinned uni2ts cfd46d45. Its forward defines the quantile recurrence reproduced here.
_FORWARD_SHA256 = 'eb826593fd54756927e4e15a5bab8dce97c43e905ca8dc2ce8520bea82d2a086'


@dataclass
class LengthGroup:
    indices: tuple[int, ...]
    context_tokens: int
    prediction_tokens: int
    inputs: tuple
    output: torch.Tensor


@dataclass
class MoiraiBlockState:
    rows: int
    context_length: int
    horizon: int
    groups: tuple[LengthGroup, ...]
    blocks_done: int = 0
    cache_bytes = 0
    cache_position = None

    @property
    def bucket(self):
        # Even equal-shape *additional* rows can change GEMM rounding enough to
        # diverge after 16 autoregressive rounds. Preserve each native batch.
        return (id(self),)

    @property
    def tensor_bytes(self):
        sizes = {}
        for group in self.groups:
            for tensor in (*group.inputs, group.output):
                storage = tensor.untyped_storage()
                sizes[storage.data_ptr()] = storage.nbytes()
        return sum(sizes.values())


class MoiraiStepper:
    block_size = OUTPUT_BLOCK_SIZE

    def __init__(self, runtime):
        from vtsfm.compilation import execution_mode

        if (runtime.precision != 'fp32' or runtime.attention_mode != 'upstream'
                or execution_mode() != 'eager' or os.environ.get('NVIDIA_TF32_OVERRIDE') == '1'
                or torch.backends.cuda.matmul.allow_tf32):
            raise ValueError('Moirai block mode requires eager FP32 and upstream attention')
        if hashlib.sha256(inspect.getsource(Moirai2Forecast.forward).encode()).hexdigest() != _FORWARD_SHA256:
            raise RuntimeError('Moirai forecast implementation changed; revalidate the pinned block recurrence')
        model = runtime.model
        if (model.patch_size != PATCH_SIZE or model.num_predict_token * PATCH_SIZE != self.block_size
                or tuple(model.quantile_levels) != tuple(i / 10 for i in range(1, 10))):
            raise ValueError('Moirai block mode requires the pinned 16-point patch and nine-quantile decoder')
        self.runtime = runtime
        self.quantiles = None

    @torch.inference_mode()
    def initialize(self, context, horizon, *, context_lengths=None):
        model_device = next(self.runtime.model.parameters()).device
        if context.ndim != 2 or context.dtype != torch.float32 or context.device != model_device:
            raise ValueError('Moirai block input must be FP32 [rows, context] on the model device')
        rows, width = context.shape
        if (not 1 <= rows <= DEFAULT_LIMITS.max_request_rows or not 1 <= width <= DEFAULT_LIMITS.max_context
                or type(horizon) is not int or not 1 <= horizon <= DEFAULT_LIMITS.max_horizon):
            raise ValueError('Moirai block shape exceeds the admission envelope')
        if context_lengths is None:  # Direct numerical callers lack the transport's CPU lengths.
            context_lengths = torch.isfinite(context).sum(dim=1).tolist()
        if (len(context_lengths) != rows or any(type(n) is not int or not 1 <= n <= width for n in context_lengths)):
            raise ValueError('Moirai block lengths must match the numerical rows')
        lengths = torch.tensor(context_lengths, device=context.device)
        padding = torch.arange(width, device=context.device) < (width - lengths[:, None])
        if not torch.where(padding, context.isnan(), context.isfinite()).all():
            raise ValueError('Moirai block rows require NaN left padding and finite observed histories')

        groups = []
        for length in sorted(set(context_lengths)):
            indices = tuple(i for i, n in enumerate(context_lengths) if n == length)
            # Uniform requests (including the eight-row maximum) need no gather.
            values = context[:, -length:] if len(indices) == rows else context[list(indices), -length:]
            key = (length, horizon, context.device)
            cached = self.runtime._cached_runner
            if cached is not None and cached[0] == key and cached[1].module is self.runtime.model:
                runner = cached[1]
            else:
                runner = self.runtime._forecast_runner(key, cached)
            if self.runtime.model.training:
                runner.eval()
            inputs = runner._convert(PATCH_SIZE, values.unsqueeze(-1),
                                     torch.ones_like(values, dtype=torch.bool).unsqueeze(-1),
                                     torch.zeros_like(values, dtype=torch.bool))
            n = runner.prediction_token_length(PATCH_SIZE)
            output = values.new_empty((len(indices), n, 9, PATCH_SIZE))
            groups.append(LengthGroup(indices, runner.context_token_length(PATCH_SIZE), n, inputs, output))
        if self.quantiles is None or self.quantiles.device != context.device:
            self.quantiles = context.new_tensor(self.runtime.model.quantile_levels)
        return MoiraiBlockState(rows, width, horizon, tuple(groups))

    @torch.inference_mode()
    def prefill(self, states):
        self._step(states, recursive=False)

    @torch.inference_mode()
    def advance(self, states):
        self._step(states, recursive=True)

    def _step(self, states, *, recursive):
        for state in states:
            if not state.groups or self.complete(state) or recursive != (state.blocks_done > 0):
                raise ValueError('Moirai forecast stage does not match live request progress')
            for group in state.groups:
                self._advance_group(group, state.blocks_done)
            state.blocks_done += 1

    def _advance_group(self, group, round_index):
        predictions = self.runtime.model(*group.inputs, training_mode=False)
        per_call = self.runtime.model.num_predict_token
        start = round_index * per_call
        count = min(per_call, group.prediction_tokens - start)
        absolute = group.context_tokens + start
        # Shape [rows, (previous trajectories), future patches, quantiles, patch].
        block = predictions[..., absolute - 1, :].unflatten(-1, (per_call, 9, PATCH_SIZE))[..., :count, :, :]
        if round_index:
            candidates = block.permute(0, 2, 1, 3, 4).flatten(2, 3)
            block = torch.quantile(candidates, self.quantiles, dim=2).permute(1, 2, 0, 3)
        group.output[:, start:start + count].copy_(block)
        if start + count == group.prediction_tokens:
            group.inputs = ()
            return
        if not round_index:
            # Only targets and prediction masks are modified by future rounds.
            expanded = tuple(field.unsqueeze(1).expand(-1, 9, *field.shape[1:]) for field in group.inputs)
            group.inputs = (expanded[0].clone(), *expanded[1:5], expanded[5].clone())
        group.inputs[0][:, :, absolute:absolute + count].copy_(block.transpose(1, 2))
        group.inputs[5][:, :, absolute:absolute + count] = False

    def model_call_work(self, states):
        return sum(len(group.indices) * (9 if state.blocks_done else 1)
                   * (group.context_tokens + group.prediction_tokens)**2
                   for state in states for group in state.groups)

    def complete(self, state):
        return state.blocks_done * self.block_size >= state.horizon

    def finalize(self, state):
        if not self.complete(state) or not state.groups:
            raise ValueError('Cannot finalize an unfinished or released Moirai forecast')
        if len(state.groups) == 1:
            values = state.groups[0].output.transpose(1, 2).flatten(2)[..., :state.horizon]
            return values.clone(memory_format=torch.contiguous_format)
        out = state.groups[0].output.new_empty((state.rows, 9, state.horizon))
        for group in state.groups:
            out[list(group.indices)] = group.output.transpose(1, 2).flatten(2)[..., :state.horizon]
        return out

    def release(self, state):
        state.groups = ()
