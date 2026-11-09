"""Tensor-only forecasting around the pinned TimesFM 2.5 decoder."""

import torch
import timesfm
from timesfm.torch.util import revin
from torch.nn import functional as F

from vtsfm.execution import ExecutionResult, PreparedBatch
from vtsfm.compilation import configure_compilation

from .spec import FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, TIMESFM25


class TimesFMRuntime:
    def __init__(self, config):
        self.model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False).model
        configure_compilation(self, 'timesfm25')

    @torch.inference_mode()
    def forecast(self, context, horizon):
        """Forecast [rows, 9, horizon] from finite histories with left NaN padding."""
        context = context.to(device=next(self.model.parameters()).device, dtype=torch.float32)
        return self._forecast(context, torch.isfinite(context), horizon)

    @torch.inference_mode()
    def execute(self, batch: PreparedBatch) -> ExecutionResult:
        """Collate prepared independent rows with a common dense history length."""
        key = batch.plan.key
        if (key.adapter, key.model_id, key.model_revision) != ('timesfm25', MODEL_ID, MODEL_REVISION):
            raise ValueError('Batch key does not match the TimesFM checkpoint')
        capabilities = TIMESFM25.execution_capabilities
        capabilities.validate_key(key)
        if batch.requests[0].capabilities != capabilities:
            raise ValueError('Prepared batch must use the TimesFM execution capabilities')
        device = next(self.model.parameters()).device
        if batch.requests[0].context.device != device:
            raise ValueError('Prepared TimesFM inputs must be on the model device')

        # A wider padded row changes upstream normalization even if its mask
        # hides the added points. Use the same history width as isolated forecasts.
        length = batch.requests[0].context_lengths[0]
        if any(n != length for request in batch.requests for n in request.context_lengths):
            raise ValueError('Prepared TimesFM batches require equal history lengths')
        context = torch.cat([request.context[:, -length:] for request in batch.requests])
        observed = torch.cat([request.observed_mask[:, -length:] for request in batch.requests])
        if not (observed & torch.isfinite(context)).all():
            raise ValueError('Prepared TimesFM histories must be finite and fully observed after left padding')
        quantiles = self._forecast(context, observed, key.prediction_length)
        return ExecutionResult(batch, torch.stack([
            quantiles[:, FORECAST_SETTINGS.quantiles.index(level)] for level in key.quantile_levels
        ], dim=1))

    def _decode_quantiles(self, context, padding, horizon):
        prefill, spread, autoregressive = self.model.decode(horizon, context, padding)
        blocks = [prefill[:, -1]]
        if autoregressive is not None:
            blocks.append(autoregressive.flatten(1, 2))
        # Channel zero is the point head. The public API returns quantiles 1–9.
        return torch.cat(blocks, dim=1)[:, :horizon, 1:], spread[:, :horizon, 1:]

    def _forecast(self, context, observed, horizon):
        if not 1 <= horizon <= self.model.os:
            raise ValueError(f'TimesFM forecast horizon must be in [1,{self.model.os}]')
        # Follow the pinned Google TimesFM 2.5 forecast closure (Apache-2.0).
        # Its outer normalization includes zero left padding to the next patch.
        left = -context.shape[-1] % self.model.p
        values = F.pad(context.masked_fill(~observed, 0), (left, 0))
        padding = F.pad(~observed, (left, 0), value=True)
        positive = (values >= 0).all(dim=-1, keepdim=True).unsqueeze(-1)
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True)
        normalized = revin(values, mean, std, reverse=False)

        forecast, spread = self._decode_quantiles(normalized, padding, horizon)
        flipped_forecast, flipped_spread = self._decode_quantiles(-normalized, padding, horizon)
        forecast = (forecast - flipped_forecast.flip(-1)) / 2
        spread = (spread - flipped_spread.flip(-1)) / 2

        median = forecast[..., 4:5]
        continuous = spread - spread[..., 4:5] + median
        # Enforce monotonic quantiles by clipping outward from the fixed median.
        lower = torch.cat((median, continuous[..., :4].flip(-1)), dim=-1)
        upper = torch.cat((median, continuous[..., 5:]), dim=-1)
        lower = lower.cummin(dim=-1).values[..., 1:].flip(-1)
        upper = upper.cummax(dim=-1).values[..., 1:]
        quantiles = revin(torch.cat((lower, median, upper), dim=-1), mean, std, reverse=True)
        return torch.where(positive, quantiles.clamp_min(0), quantiles).transpose(1, 2)
