"""Scoped native GEMM geometry for independent Toto calibration origins.

Imported only for an actual multi-origin forecast. Larger FP32 GEMMs can round
differently from native single-origin calls, which affects small residuals.
"""

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


class _OriginLinearMode(TorchFunctionMode):
    """Keep each origin's contiguous linear input without changing the model.

    Toto retains origins along dimension zero, including variate layers that
    flatten it with time. Splitting equally restores each native GEMM shape.
    The mode stack is thread-local; other calls and threads retain their paths.
    """

    def __init__(self, num_origins):
        super().__init__()
        if type(num_origins) is not int or num_origins < 2:
            raise ValueError('num_origins must be an integer of at least two')
        self.num_origins = num_origins

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is not F.linear:
            return func(*args, **kwargs)
        if torch.is_grad_enabled():
            raise RuntimeError('Toto calibration linear mode is inference-only')
        inputs = args[0] if args else kwargs['input']
        if (inputs.ndim < 2 or inputs.shape[0] == 0
                or inputs.shape[0] % self.num_origins):
            raise ValueError('Linear inputs must preserve independent origin groups')
        outputs = []
        for part in inputs.split(inputs.shape[0] // self.num_origins, dim=0):
            # Calls inside this handler forward to the next mode, not this one.
            if args:
                outputs.append(func(part.contiguous(), *args[1:], **kwargs))
            else:
                outputs.append(func(**{**kwargs, 'input': part.contiguous()}))
        return torch.cat(outputs, dim=0)
