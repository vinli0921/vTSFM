"""Tensor forecasting around the official Moirai 2.0 recursive quantile decoder."""

import os
import sys

import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from vtsfm.compilation import compile_core, execution_mode
from vtsfm.execution import ExecutionResult, PreparedBatch
from vtsfm.precision import PRECISION_ENV, configure_precision

from .spec import FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, MOIRAI2, resolve_attention_config


class _InferenceMoiraiForecast(Moirai2Forecast):
    def save_hyperparameters(self, *args, **kwargs):
        # Lightning 2.6.6 otherwise keeps inspect.currentframe() in that frame's
        # own locals, forming a cycle that also pins our caller's CUDA tensors.
        # Supply the same constructor frame directly, preserving upstream
        # hyperparameter collection without waiting for cyclic GC to free IO.
        if kwargs.get('frame') is None:
            kwargs['frame'] = sys._getframe(1)
        return super().save_hyperparameters(*args, **kwargs)


class MoiraiRuntime:
    def __init__(self, config):
        mode = execution_mode()
        self.attention_mode, precision = resolve_attention_config(mode, os.environ.get(PRECISION_ENV))
        self.precision = configure_precision(MOIRAI2.name, precision)
        self.model = Moirai2Module(**config)
        if self.precision == 'bf16':
            from .precision import prepare_native_bf16

            self._precision_handles = prepare_native_bf16(self.model)
        elif self.precision == 'tf32':
            from .precision import prepare_native_tf32

            prepare_native_tf32(self.model)
        # Opt-in (#24): exact fused attention for univariate rows. Every request
        # this runtime accepts is univariate (target_dim=1; execute() rejects
        # grouped rows), and the patch raises on multivariate input anyway.
        if self.attention_mode == 'univariate':
            from .attention import install_univariate_attention

            install_univariate_attention(self.model)
        if mode == 'compile':
            compile_model(self.model)
        # Publish both cache entries together: (primary key, runner, previous).
        # The previous entry is just (key, runner), so retention ends at two.
        self._cached_runner = None

    @torch.inference_mode()
    def forecast(self, context, horizon):
        """Live-serving path: FP32 [rows, width] with left NaN padding -> [rows, 9, horizon]."""
        context = context.to(device=next(self.model.parameters()).device, dtype=torch.float32)
        observed = torch.isfinite(context)
        # Request parsing rejects non-finite targets, so NaN marks only left padding.
        if self.precision in ('bf16', 'tf32'):
            # Match native predict's per-length batches: mixing padded widths
            # can change reduced-precision CUDA rounding even with equivalent masks. Only
            # these integer lengths cross to the host; histories stay on device.
            lengths = observed.sum(-1).tolist()
            return self._forecast_by_length(context, observed, ~observed, horizon, lengths)
        return self._forecast(context, observed, ~observed, horizon)

    @torch.inference_mode()
    def execute(self, batch: PreparedBatch) -> ExecutionResult:
        """Forecast prepared rows, grouping BF16/TF32 rows by their known history lengths.

        Returns [rows, requested quantiles, horizon] in the original row order.
        """
        key = batch.plan.key
        if (key.adapter, key.model_id, key.model_revision) != (MOIRAI2.name, MODEL_ID, MODEL_REVISION):
            raise ValueError('Batch key does not match the Moirai checkpoint')
        capabilities = MOIRAI2.execution_capabilities
        capabilities.validate_key(key)
        # Equal capabilities also exclude grouped rows (multivariate input).
        if batch.requests[0].capabilities != capabilities:
            raise ValueError('Prepared batch must use the Moirai execution capabilities')
        device = next(self.model.parameters()).device
        if batch.requests[0].context.device != device:
            raise ValueError('Prepared Moirai inputs must be on the model device')

        # Crop extra left padding. Fully padded patches would not change the
        # forecast, but they still cost attention and scaler work.
        lengths = [n for request in batch.requests for n in request.context_lengths]
        width = max(lengths)
        context = torch.cat([request.context[:, -width:] for request in batch.requests])
        observed = torch.cat([request.observed_mask[:, -width:] for request in batch.requests])
        # Padding comes from the right-aligned lengths, not the mask, so it
        # stays distinct from missing observations inside a history.
        starts = width - torch.tensor(lengths, device=device)
        padding = torch.arange(width, device=device) < starts.unsqueeze(-1)
        # Upstream predict imputes interior gaps with a causal mean that reaches
        # the input tokens. That is not reproduced yet, so reject gaps. This is
        # one scalar sync, not a host copy of the inputs.
        if not (padding | (observed & torch.isfinite(context))).all():
            raise ValueError('Moirai execution requires finite, fully observed histories after left padding')

        if self.precision in ('bf16', 'tf32'):
            quantiles = self._forecast_by_length(context, observed, padding, key.prediction_length, lengths)
        else:
            quantiles = self._forecast(context, observed, padding, key.prediction_length)
        channels = [FORECAST_SETTINGS.quantiles.index(level) for level in key.quantile_levels]
        return ExecutionResult(batch, quantiles[:, channels])

    def _forecast_by_length(self, context, observed, padding, horizon, lengths):
        """Keep native batch shapes, using CPU lengths and device tensors."""
        unique_lengths = sorted(set(lengths))
        if len(unique_lengths) == 1:
            length = unique_lengths[0]
            return self._forecast(context[:, -length:], observed[:, -length:], padding[:, -length:], horizon)
        result = context.new_empty((context.shape[0], len(FORECAST_SETTINGS.quantiles), horizon))
        # BF16/TF32 runs one forecast per distinct length. In a repeated cycle
        # of three or more lengths, each runner is evicted before its next use.
        for length in unique_lengths:
            rows = [index for index, value in enumerate(lengths) if value == length]
            result[rows] = self._forecast(context[rows, -length:], observed[rows, -length:],
                                          padding[rows, -length:], horizon)
        return result

    def _forecast(self, context, observed, padding, horizon):
        """[rows, width] values, observed mask and padding mask on the model device
        -> [rows, 9, horizon] in checkpoint quantile order.

        Moirai2Forecast.forward owns patching, sample/time IDs, normalization,
        the recursive quantile loop, and cropping to the horizon. This replaces
        only upstream predict's NumPy preprocessing, whose ragged left padding
        repeats each history's first value as observed.
        """
        # Zero matches what upstream gives an isolated history's partial patch,
        # and keeps NaN out of the scaler, where NaN * False would still be NaN.
        target = context.masked_fill(~observed, 0)
        key = (context.shape[-1], horizon, context.device)
        cached = self._cached_runner
        if cached is not None and cached[0] == key and cached[1].module is self.model:
            runner = cached[1]
        else:
            runner = self._forecast_runner(key, cached)
        if self.model.training:
            runner.eval()
        # Upstream returns [rows, 9 quantiles, horizon] for univariate targets.
        quantiles = runner(past_target=target.unsqueeze(-1), past_observed_target=observed.unsqueeze(-1),
                           past_is_pad=padding)
        # A one-patch result can be a contiguous view into the full-context
        # quantile grid; longer crops may retain patch-rounded predictions.
        # Own exactly the public bytes used by shared resident accounting.
        return quantiles.clone(memory_format=torch.contiguous_format)

    def _forecast_runner(self, key, cached):
        """On a primary miss, promote the previous shape or build a new runner."""
        width, horizon, device = key
        previous = None
        if cached is not None and cached[0][2] == device and cached[1].module is self.model:
            previous = (cached[0], cached[1])
            secondary = cached[2]
            if secondary is not None and secondary[0] == key:
                runner = secondary[1]
                self._cached_runner = (key, runner, previous)
                return runner

        runner = _InferenceMoiraiForecast(
            module=self.model, prediction_length=horizon, context_length=width,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        ).to(device).eval()
        self._cached_runner = (key, runner, previous)
        return runner


def compile_model(model):
    """Compile the encoder; keep scaling and recursive quantiles upstream/eager."""
    from uni2ts.module.attention import GroupedQueryAttention
    from uni2ts.module.ffn import FeedForward
    from uni2ts.module.norm import RMSNorm
    from uni2ts.module.position.attn_projection import QueryKeyProjection

    native_calls = []
    for module in model.encoder.modules():
        if isinstance(module, QueryKeyProjection):
            # Static Python shape metadata: avoid cached-property locks in tracing.
            _ = module.proj_width, module.split_sizes
        if isinstance(module, (RMSNorm, GroupedQueryAttention)):
            native_calls.append((module, 'forward'))
        if isinstance(module, FeedForward):
            native_calls.append((module, 'activation'))
    compile_core(model.encoder, native_calls)
