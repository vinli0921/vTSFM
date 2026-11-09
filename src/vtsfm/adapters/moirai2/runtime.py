"""Official Moirai 2.0 recursive quantile inference."""

import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from vtsfm.compilation import configure_compilation


class MoiraiRuntime:
    def __init__(self, config):
        self.model = Moirai2Module(**config)
        configure_compilation(self, 'moirai2')

    def forecast(self, context, horizon):
        # The official predict API takes NumPy histories and returns NumPy forecasts.
        inputs = [row[torch.isfinite(row)].cpu().numpy() for row in context]
        runner = Moirai2Forecast(
            module=self.model, prediction_length=horizon, context_length=max(map(len, inputs)),
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        ).to(next(self.model.parameters()).device).eval()
        return torch.from_numpy(runner.predict(inputs))
