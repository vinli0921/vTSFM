"""Tensor forecasting around the official Toto 2 decoder."""

import os

import torch
from torch.nn import functional as F
from toto2 import Toto2Model
from toto2.configuration import Toto2ModelConfig

from vtsfm.execution import ExecutionResult, PreparedBatch
from vtsfm.precision import configure_precision
from vtsfm.compilation import compile_core, compile_method, execution_mode

from . import fast_scaling
from .spec import DECODE_BLOCK_SIZE, FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, TOTO2, compile_scope


class TotoRuntime:
    def __init__(self, config):
        mode = execution_mode()
        scope = compile_scope(os.environ, mode, os.environ.get('VTSFM_PRECISION'))
        self.precision = configure_precision('toto2', 'fp32' if scope == 'whole' else None)
        self.model = Toto2Model(Toto2ModelConfig(**config))
        if self.precision == 'bf16':
            from dd_unit_scaling import RMSNorm
            from toto2.model import RotaryProjection, SelfAttention
            from .precision import prepare_native_bf16

            self._precision_handles = prepare_native_bf16(
                self.model, RMSNorm, RotaryProjection, attention_type=SelfAttention)
        else:
            # FP32 and TF32: vLLM builds the model on CUDA, whose sin/cos can differ from the official CPU-built rotary tables by one ulp (#85).
            from toto2.model import RotaryProjection
            from .precision import prepare_native_rotary_tables

            prepare_native_rotary_tables(self.model, RotaryProjection)
        # Inference-only unit-scaling shims (#74); set
        # VTSFM_TOTO_FAST_UNIT_SCALING=0 to keep the upstream autograd wrappers.
        self.fast_unit_scaling = os.environ.get('VTSFM_TOTO_FAST_UNIT_SCALING', '1') != '0'
        if self.fast_unit_scaling:
            fast_scaling.install()
        if scope == 'whole':
            # Both single-request forecast() and cross-request execute() call this
            # tensor boundary, including mask preparation and recursive decoding.
            compile_method(self, '_forecast')
        elif mode == 'compile':
            compile_model(self.model)

    @torch.inference_mode()
    def forecast(self, context, horizon):
        context = context.to(device=next(self.model.parameters()).device, dtype=torch.float32)
        return self._forecast(context, torch.isfinite(context), horizon)

    @torch.inference_mode()
    def execute(self, batch: PreparedBatch) -> ExecutionResult:
        """Forecast all prepared rows in one call; returns [rows, requested quantiles, horizon]."""
        key = batch.plan.key
        if (key.adapter, key.model_id, key.model_revision) != (TOTO2.name, MODEL_ID, MODEL_REVISION):
            raise ValueError('Batch key does not match the Toto checkpoint')
        capabilities = TOTO2.execution_capabilities
        capabilities.validate_key(key)
        if batch.requests[0].capabilities != capabilities:
            raise ValueError('Prepared batch must use the Toto execution capabilities')
        # Multi-row groups would let Toto's variate attention mix independent series.
        if any(size != 1 for request in batch.requests for size in request.metadata.group_sizes):
            raise ValueError('Toto execution requires independent single-row groups')
        if batch.requests[0].context.device != next(self.model.parameters()).device:
            raise ValueError('Prepared Toto inputs must be on the model device')

        # Crop extra left padding so a wider prepared width matches the isolated forecast.
        length = max(n for request in batch.requests for n in request.context_lengths)
        context = torch.cat([request.context[:, -length:] for request in batch.requests])
        observed = torch.cat([request.observed_mask[:, -length:] for request in batch.requests])
        if not torch.isfinite(context[observed]).all():
            raise ValueError('Observed Toto history values must be finite')

        quantiles = self._forecast(context, observed, key.prediction_length)
        channels = [FORECAST_SETTINGS.quantiles.index(level) for level in key.quantile_levels]
        return ExecutionResult(batch, quantiles[:, channels])

    def _forecast(self, context, observed, horizon):
        """[rows, width] values and mask on the model device -> [rows, 9, horizon]."""
        left = -context.shape[-1] % self.model.config.patch_size
        # Zero unobserved values like upstream's nan_to_num inputs; Toto forces the
        # final context patch to observed, so filler there would leak in otherwise.
        target = F.pad(context.masked_fill(~observed, 0), (left, 0))
        mask = F.pad(observed, (left, 0), value=False)
        # One variate per batch element keeps every row an independent series.
        inputs = {'target': target.unsqueeze(1), 'target_mask': mask.unsqueeze(1),
                  'series_ids': torch.zeros((len(target), 1), dtype=torch.long, device=target.device)}
        with fast_scaling.fast_unit_scaling(self.fast_unit_scaling):
            quantiles = self.model.forecast(inputs, horizon=horizon, decode_block_size=DECODE_BLOCK_SIZE)
        # Upstream returns [9 quantiles, rows, variates, horizon].
        # Public transport is FP32 and owns only the requested horizon, even
        # when the upstream crop is already contiguous or shares padded storage.
        output = quantiles[:, :, 0, :].permute(1, 0, 2)
        return output.to(dtype=torch.float32, memory_format=torch.contiguous_format, copy=True)


def compile_model(model):
    """Compile the transformer, retaining native unit-scaling and attention math."""
    from dd_unit_scaling import RMSNorm
    from toto2.model import GatedLinearUnitFeedForwardNetwork, SelfAttention

    native_calls = [(module, 'forward') for module in model.transformer.modules()
                    if isinstance(module, (RMSNorm, SelfAttention, GatedLinearUnitFeedForwardNetwork))]
    changed_layers = []
    try:
        if any(parameter.dtype in (torch.bfloat16, torch.float16)
               for parameter in model.transformer.parameters()):
            for layer in model.transformer.layers:
                original, local = layer.forward, 'forward' in vars(layer)
                layer.forward = _native_residual_forward(layer)
                changed_layers.append((layer, original, local))
        compile_core(model.transformer, native_calls)
    except BaseException:
        for layer, original, local in reversed(changed_layers):
            if local:
                layer.forward = original
            else:
                del layer.forward
        raise


def _native_residual_forward(layer):
    # Derived from Datadog Toto's SelfAttentionTransformerLayer.forward,
    # Copyright 2026 Datadog, Inc., Apache-2.0 (pinned upstream in spec.py).
    # Only residual_add stays native: fusion otherwise changes BF16 rounding.
    from dd_unit_scaling import functional as U

    residual_add = torch.compiler.disable(U.residual_add, recursive=True)

    def forward(x, seq_ids=None, **kwargs):
        x, skip = U.residual_split(x, layer.attn_tau)
        x = layer.attn(layer.norm1(x), seq_ids, **kwargs)
        x = residual_add(layer.attn_resid_dropout(x), skip, layer.attn_tau)

        x, skip = U.residual_split(x, layer.mlp_tau)
        x = layer.ffn(layer.norm2(x))
        return residual_add(layer.mlp_resid_dropout(x), skip, layer.mlp_tau)

    return forward
