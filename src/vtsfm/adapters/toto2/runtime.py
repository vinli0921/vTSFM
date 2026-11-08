"""Tensor forecasting around the official Toto 2 decoder."""

import torch
from torch.nn import functional as F
from toto2 import Toto2Model
from toto2.configuration import Toto2ModelConfig

from vtsfm.execution import ExecutionResult, PreparedBatch

from .spec import DECODE_BLOCK_SIZE, FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, TOTO2


class TotoRuntime:
    def __init__(self, config):
        self.model = Toto2Model(Toto2ModelConfig(**config))

    @torch.inference_mode()
    def forecast(self, context, horizon):
        context = context.to(next(self.model.parameters()).device)
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
        quantiles = self.model.forecast(inputs, horizon=horizon, decode_block_size=DECODE_BLOCK_SIZE)
        # Upstream returns [9 quantiles, rows, variates, horizon].
        return quantiles[:, :, 0, :].permute(1, 0, 2)
