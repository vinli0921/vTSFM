"""Official Toto 2 block decoding for independent univariate series."""

import torch
from torch.nn import functional as F
from toto2 import Toto2Model
from toto2.configuration import Toto2ModelConfig


class TotoRuntime:
    def __init__(self, config):
        self.model = Toto2Model(Toto2ModelConfig(**config))

    def forecast(self, context, horizon):
        context = context.to(next(self.model.parameters()).device)
        context = F.pad(context, (-context.shape[-1] % self.model.config.patch_size, 0), value=float('nan'))
        mask = torch.isfinite(context).unsqueeze(1)
        target = torch.nan_to_num(context).unsqueeze(1)
        inputs = {'target': target, 'target_mask': mask,
                  'series_ids': torch.zeros((len(context), 1), dtype=torch.long, device=context.device)}
        quantiles = self.model.forecast(inputs, horizon=horizon, decode_block_size=768)
        return quantiles[:, :, 0, :].permute(1, 0, 2)
