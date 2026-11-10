"""Keep native per-origin FP32 reductions and projections during calibration.

Loaded sktime pipelines are shared. A thread-local TorchFunctionMode confines
these operation shapes to one forecast call without changing cached modules.
"""

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


class ChronosOriginMode(TorchFunctionMode):
    def __init__(self, pipeline, context_widths):
        super().__init__()
        self.context_widths = tuple(context_widths)
        self.origins = len(self.context_widths)
        self.context_limit = pipeline.model_context_length
        self.prediction_length = pipeline.model_prediction_length
        self.reductions = 0
        # Group attention transposes [origin, time, feature] before projection.
        self.axes = {id(module.weight): (1 if '.layer.1.' in name else 0)
                     for name, module in pipeline.model.named_modules()
                     if isinstance(module, torch.nn.Linear)}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is F.linear:
            inputs = args[0] if args else kwargs['input']
            weight = args[1] if len(args) > 1 else kwargs['weight']
            axis = self.axes.get(id(weight))
            if axis is None:
                return func(*args, **kwargs)
            if torch.is_grad_enabled():
                raise RuntimeError('Calibration execution is inference-only')
            if inputs.shape[axis] % self.origins:
                raise ValueError('Chronos projections must preserve origin groups')
            values = []
            for part in inputs.chunk(self.origins, dim=axis):
                if args:
                    values.append(func(part.contiguous(), *args[1:], **kwargs))
                else:
                    values.append(func(**{**kwargs, 'input': part.contiguous()}))
            return torch.cat(values, dim=axis)
        if func not in (torch.nanmean, torch.Tensor.nanmean):
            return func(*args, **kwargs)

        inputs = args[0] if args else kwargs['input']
        if inputs.ndim != 2 or inputs.shape[0] % self.origins:
            raise ValueError('Chronos normalization must preserve origin groups')
        # Native InstanceNorm makes two reductions per forecast step. Long
        # horizons append one native prediction block to every quantile path.
        step = self.reductions // 2
        self.reductions += 1
        widths = [min(width + step * self.prediction_length, self.context_limit)
                  for width in self.context_widths]
        if inputs.shape[-1] != max(widths):
            raise ValueError('Chronos normalization context differs from its native schedule')
        values = []
        for width, part in zip(widths, inputs.chunk(self.origins, dim=0)):
            # A contiguous view may retain a misaligned storage offset. A fresh
            # allocation also preserves the native reduction's alignment.
            part = part[..., -width:].clone(memory_format=torch.contiguous_format)
            if args:
                values.append(func(part, *args[1:], **kwargs))
            else:
                values.append(func(**{**kwargs, 'input': part}))
        return torch.cat(values, dim=0)
