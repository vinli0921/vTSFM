"""Tensor-only forecasting around the pinned TimesFM 2.5 decoder."""

import os

import torch
import timesfm
from timesfm.torch.util import revin
from torch.nn import functional as F

from vtsfm.execution import ExecutionResult, PreparedBatch
from vtsfm.precision import PRECISION_ENV, configure_precision
from vtsfm.compilation import compile_core, compile_method, execution_mode
from vtsfm.config import FORECAST_BATCH_ROWS_ENV
from vtsfm.stepping import CONTINUOUS_ENV

from .graph_config import FORECAST_MODE_ENV, validate_fixed_forecast_execution
from .spec import FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, TIMESFM25, compile_scope, flip_layout


# Keep explicit graph replay under the adapter’s separate buffer owner.
_WIDER_INDUCTOR_OPTIONS = {'triton.cudagraphs': False}


class TimesFMRuntime:
    def __init__(self, config):
        execution = execution_mode()
        scope = compile_scope(os.environ, execution, os.environ.get(PRECISION_ENV),
                              continuous=os.environ.get(CONTINUOUS_ENV) == '1')
        if FORECAST_MODE_ENV in os.environ:
            validate_fixed_forecast_execution(os.environ.get(PRECISION_ENV), execution)
            if scope != 'core':
                raise ValueError('TimesFM positioned forecasts require the standard whole-forecast serving path')
        self.compile_scope = 'eager' if scope == 'core' and execution == 'eager' else scope
        self.flip_layout = flip_layout(os.environ, os.environ.get(PRECISION_ENV),
            independent_whole_forecast=(
                FORECAST_MODE_ENV not in os.environ and os.environ.get(CONTINUOUS_ENV) != '1'
                and os.environ.get(FORECAST_BATCH_ROWS_ENV, '0') == '0'
                and scope not in ('positioned', 'positioned-eager')
            ))
        self.precision = configure_precision(
            'timesfm25', 'fp32' if scope != 'core' or self.flip_layout == 'split' else None)
        self.model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False).model
        self.fixed_forecast = None
        self.positioned_forecast = None
        if self.compile_scope == 'core':
            compile_model(self.model)
        elif self.compile_scope == 'wide':
            compile_method(self.model, 'forward', inductor_options=_WIDER_INDUCTOR_OPTIONS)
        elif self.compile_scope == 'whole':
            compile_method(self, '_forecast', inductor_options=_WIDER_INDUCTOR_OPTIONS)
        elif self.compile_scope in ('positioned-eager', 'positioned'):
            from .positioned_compile import PositionedForecast

            self.positioned_forecast = PositionedForecast(self, compiled=self.compile_scope == 'positioned')


    @torch.inference_mode()
    def forecast(self, context, horizon):
        """Forecast [rows, 9, horizon] from finite histories with left NaN padding."""
        context = context.to(device=next(self.model.parameters()).device, dtype=torch.float32)
        policy = getattr(self, 'fixed_forecast', None)
        if policy is not None:
            output = policy.try_forecast(context, horizon)
            if output is not None:
                return output
        return self._forecast(context, torch.isfinite(context), horizon)

    @torch.inference_mode()
    def execute(self, batch: PreparedBatch) -> ExecutionResult:
        """Collate prepared independent rows with a common dense history length."""
        if getattr(self, 'flip_layout', 'packed') == 'split' and len(batch.requests) > 1:
            raise ValueError('TimesFM split flips require one independent forecast request per model call')
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
        if getattr(self.model, '_vtsfm_paged_stepper_mode', None) == 'patched':
            raise ValueError('TimesFM model is owned by a paged stepper. Close it before running a whole forecast.')
        if not 1 <= horizon <= self.model.os:
            raise ValueError(f'TimesFM forecast horizon must be in [1,{self.model.os}]')
        positioned = getattr(self, 'positioned_forecast', None)
        if positioned is not None:
            return positioned.run(context, observed, horizon)
        # Follow the pinned Google TimesFM 2.5 forecast closure (Apache-2.0).
        # Its outer normalization includes zero left padding to the next patch.
        left = -context.shape[-1] % self.model.p
        values = F.pad(context.masked_fill(~observed, 0), (left, 0))
        padding = F.pad(~observed, (left, 0), value=True)
        positive = (values >= 0).all(dim=-1, keepdim=True).unsqueeze(-1)
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True)
        normalized = revin(values, mean, std, reverse=False)

        if getattr(self, 'flip_layout', 'packed') == 'split':
            # Use the same batch size as upstream for both x and -x. Packing
            # the passes can change FP32 reductions in geometry-sensitive GEMMs.
            forecast, spread = self._decode_quantiles(normalized, padding, horizon)
            flipped_forecast, flipped_spread = self._decode_quantiles(-normalized, padding, horizon)
        else:
            rows = normalized.shape[0]
            both, both_spread = self._decode_quantiles(
                torch.cat((normalized, -normalized)), padding.repeat(2, 1), horizon)
            forecast, flipped_forecast = both[:rows], both[rows:]
            spread, flipped_spread = both_spread[:rows], both_spread[rows:]
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


def compile_model(model):
    """Keep native norm, activation and SDPA arithmetic inside the neural core."""
    from timesfm.torch.normalization import RMSNorm
    from timesfm.torch.transformer import MultiHeadAttention

    native_calls = []
    for module in model.modules():
        if isinstance(module, (RMSNorm, torch.nn.ReLU, torch.nn.SiLU, torch.nn.GELU)):
            native_calls.append((module, 'forward'))
        if isinstance(module, MultiHeadAttention):
            native_calls.append((module, 'attention_fn'))
    compile_core(model, native_calls)
