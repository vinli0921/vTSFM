"""CPU-safe selection of worker-local experimental numerical precision."""

import os

from vtsfm.stepping import CONTINUOUS_ENV


PRECISION_ENV = 'VTSFM_PRECISION'
TF32_OVERRIDE_ENV = 'NVIDIA_TF32_OVERRIDE'
PRECISIONS = ('fp32', 'tf32', 'bf16')


def validate_precision(adapter: str, precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f'forecast precision must be one of {PRECISIONS}')
    if precision != 'fp32' and adapter not in ('timesfm25', 'toto2', 'moirai2'):
        raise ValueError('Experimental precision supports only numerical TSFM adapters')
    if precision == 'bf16' and adapter != 'toto2':
        raise ValueError('Native BF16 currently supports Toto only; other adapters remain FP32')
    return precision


def configure_precision(adapter: str) -> str:
    """Set a dedicated model worker's matmul policy before its first forecast.

    The server owns one adapter and precision per worker. This is not a
    per-request switch: Torch's FP32 matmul setting is process-global.
    """
    precision = validate_precision(adapter, os.environ.get(PRECISION_ENV, 'fp32'))
    if precision != 'fp32' and os.environ.get(CONTINUOUS_ENV) == '1':
        raise ValueError('Forecast-block continuous batching currently requires FP32 precision')
    expected = '1' if precision == 'tf32' else '0'
    override = os.environ.get(TF32_OVERRIDE_ENV)
    if override is not None and override != expected:
        raise ValueError(f'{TF32_OVERRIDE_ENV}={override!r} conflicts with {precision}; '
                         'start a fresh worker with matching precision settings')
    import torch

    torch.set_float32_matmul_precision('high' if precision == 'tf32' else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
    return precision
