"""Tensor forecasting around the official Moirai 2.0 recursive quantile decoder."""

import sys

import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from vtsfm.execution import ExecutionResult, PreparedBatch

from .spec import FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, MOIRAI2


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
        self.model = Moirai2Module(**config)

    @torch.inference_mode()
    def forecast(self, context, horizon):
        """Live-serving path: FP32 [rows, width] with left NaN padding -> [rows, 9, horizon]."""
        context = context.to(device=next(self.model.parameters()).device, dtype=torch.float32)
        observed = torch.isfinite(context)
        # Request parsing rejects non-finite targets, so NaN marks only left padding.
        return self._forecast(context, observed, ~observed, horizon)

    @torch.inference_mode()
    def execute(self, batch: PreparedBatch) -> ExecutionResult:
        """Forecast all prepared rows in one call; returns [rows, requested quantiles, horizon]."""
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

        quantiles = self._forecast(context, observed, padding, key.prediction_length)
        channels = [FORECAST_SETTINGS.quantiles.index(level) for level in key.quantile_levels]
        return ExecutionResult(batch, quantiles[:, channels])

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
        # A fresh wrapper per call keeps hyperparameters from leaking between
        # requests. It holds no weights; .to() moves nothing and updates the
        # device that the recursive loop uses for its quantile tensor.
        runner = _InferenceMoiraiForecast(
            module=self.model, prediction_length=horizon, context_length=context.shape[-1],
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        ).to(context.device).eval()
        # Upstream returns [rows, 9 quantiles, horizon] for univariate targets.
        quantiles = runner(past_target=target.unsqueeze(-1), past_observed_target=observed.unsqueeze(-1),
                           past_is_pad=padding)
        # A one-patch result can be a contiguous view into the full-context
        # quantile grid; longer crops may retain patch-rounded predictions.
        # Own exactly the public bytes used by shared resident accounting.
        return quantiles.clone(memory_format=torch.contiguous_format)
