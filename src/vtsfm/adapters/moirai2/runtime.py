"""Tensor forecasting around the official Moirai 2.0 recursive quantile decoder."""

import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from vtsfm.execution import ExecutionResult, PreparedBatch
from vtsfm.compilation import compile_core, execution_mode

from .spec import FORECAST_SETTINGS, MODEL_ID, MODEL_REVISION, MOIRAI2


class MoiraiRuntime:
    def __init__(self, config):
        self.model = Moirai2Module(**config)
        if execution_mode() == 'compile':
            compile_model(self.model)

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
        runner = Moirai2Forecast(
            module=self.model, prediction_length=horizon, context_length=context.shape[-1],
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        ).to(context.device).eval()
        # Upstream returns [rows, 9 quantiles, horizon] for univariate targets.
        return runner(past_target=target.unsqueeze(-1), past_observed_target=observed.unsqueeze(-1),
                      past_is_pad=padding)


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
