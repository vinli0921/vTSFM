"""Resumable eager FP32 Moirai 2 quantile forecasts for independent univariate rows.

The native module runs on the full packed sequence each round. The first call
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
class MoiraiBlockState:
    rows: int
    context_length: int
    context_tokens: int
    prediction_tokens: int
    horizon: int
    inputs: tuple
    output: torch.Tensor | None
    blocks_done: int = 0
    cache_bytes = 0
    cache_position = None

    @property
    def bucket(self):
        # Every row uses the same padded sequence geometry and prediction index.
        return (self.context_tokens, self.prediction_tokens, self.blocks_done)

    @property
    def tensor_bytes(self):
        sizes = {}
        for tensor in (*self.inputs, self.output):
            if tensor is not None:
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
    def initialize(self, context, horizon):
        model_device = next(self.runtime.model.parameters()).device
        if context.ndim != 2 or context.dtype != torch.float32 or context.device != model_device:
            raise ValueError('Moirai block input must be FP32 [rows, context] on the model device')
        rows, width = context.shape
        if (not 1 <= rows <= DEFAULT_LIMITS.max_request_rows or not 1 <= width <= DEFAULT_LIMITS.max_context
                or type(horizon) is not int or not 1 <= horizon <= DEFAULT_LIMITS.max_horizon):
            raise ValueError('Moirai block shape exceeds the admission envelope')
        key = (width, horizon, context.device)
        cached = self.runtime._cached_runner
        if cached is not None and cached[0] == key and cached[1].module is self.runtime.model:
            runner = cached[1]
        else:
            runner = self.runtime._forecast_runner(key, cached)
        if self.runtime.model.training:
            runner.eval()
        observed = torch.isfinite(context)
        inputs = runner._convert(PATCH_SIZE, context.masked_fill(~observed, 0).unsqueeze(-1),
                                 observed.unsqueeze(-1), ~observed)
        # Store the requested future only. The native prefix in quantile_prediction
        # is copied from the input but is never read by the decoder or public output.
        n = runner.prediction_token_length(PATCH_SIZE)
        out = context.new_empty((rows, n, 9, PATCH_SIZE))
        if self.quantiles is None or self.quantiles.device != context.device:
            self.quantiles = context.new_tensor(self.runtime.model.quantile_levels)
        return MoiraiBlockState(rows, width, runner.context_token_length(PATCH_SIZE), n, horizon, inputs, out)

    @torch.inference_mode()
    def prefill(self, states):
        self._step(states, recursive=False)

    @torch.inference_mode()
    def advance(self, states):
        self._step(states, recursive=True)

    def _step(self, states, *, recursive):
        first = states[0]
        if any(s.bucket != first.bucket or s.output is None or self.complete(s) for s in states):
            raise ValueError('Moirai block calls require live, compatible request states')
        if recursive != (first.blocks_done > 0):
            raise ValueError('Moirai forecast stage does not match request progress')
        if len(states) == 1:
            inputs = first.inputs
        else:
            inputs = tuple(torch.cat(field, dim=0) for field in zip(*(s.inputs for s in states), strict=True))
        predictions = self.runtime.model(*inputs, training_mode=False)
        per_call = self.runtime.model.num_predict_token
        start = first.blocks_done * per_call
        count = min(per_call, first.prediction_tokens - start)
        absolute = first.context_tokens + start
        # Select the upstream prediction source token before reshaping the head.
        # Shape: [rows, (prev trajectories), future patches, quantiles, patch].
        block = predictions[..., absolute - 1, :].unflatten(-1, (per_call, 9, PATCH_SIZE))[..., :count, :, :]
        if recursive:
            # [B,Qprev,K,Qnew,P] -> [B,K,Qprev*Qnew,P] -> [B,K,Qout,P].
            candidates = block.permute(0, 2, 1, 3, 4).flatten(2, 3)
            block = torch.quantile(candidates, self.quantiles, dim=2).permute(1, 2, 0, 3)
        for state, values in zip(states, block.split([s.rows for s in states]), strict=True):
            state.output[:, start:start + count].copy_(values)
            state.blocks_done += 1
            if self.complete(state):
                state.inputs = ()
                continue
            if not recursive:
                # Immutable masks and IDs can share their owner's base storage.
                # Target and prediction_mask are changed as forecasts become observed.
                expanded = tuple(field.unsqueeze(1).expand(-1, 9, *field.shape[1:]) for field in state.inputs)
                state.inputs = (expanded[0].clone(), *expanded[1:5], expanded[5].clone())
            state.inputs[0][:, :, absolute:absolute + count].copy_(values.transpose(1, 2))
            state.inputs[5][:, :, absolute:absolute + count] = False

    def model_call_work(self, states):
        s = states[0]
        # Model attention pair count for the active native pass. Whole-horizon
        # work is also reserved across every active owner by the V1 scheduler.
        return sum(x.rows for x in states) * (9 if s.blocks_done else 1) * (s.context_tokens + s.prediction_tokens)**2

    def complete(self, state):
        return state.blocks_done * self.block_size >= state.horizon

    def finalize(self, state):
        if not self.complete(state) or state.output is None:
            raise ValueError('Cannot finalize an unfinished or released Moirai forecast')
        return state.output.transpose(1, 2).flatten(2)[..., :state.horizon].clone(memory_format=torch.contiguous_format)

    def release(self, state):
        state.inputs = ()
        state.output = None
