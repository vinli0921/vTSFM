"""Official TimesFM 2.5 inference."""

import torch
import timesfm


class TimesFMRuntime:
    def __init__(self, config):
        self.runner = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        self.model = self.runner.model

    def forecast(self, context, horizon):
        self.model.device = next(self.model.parameters()).device
        self.model.device_count = 1
        # The official forecast API takes NumPy histories and returns NumPy forecasts.
        inputs = [row[torch.isfinite(row)].cpu().numpy() for row in context]
        self.runner.compile(timesfm.ForecastConfig(
            max_context=max(map(len, inputs)), max_horizon=horizon,
            per_core_batch_size=len(inputs), normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True,
        ))
        _, quantiles = self.runner.forecast(horizon=horizon, inputs=inputs)
        return torch.from_numpy(quantiles[..., 1:]).transpose(1, 2)
