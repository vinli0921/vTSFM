"""Univariate quantile IO helpers, with settings supplied by each adapter."""

from dataclasses import dataclass
from numbers import Real

_FP32_MAX = 3.402823466e38
_TARGET_ERROR = 'target must be a finite univariate FP32 series'


@dataclass(frozen=True)
class ForecastSettings:
    max_context: int
    max_horizon: int
    quantiles: tuple[float, ...]
    default_horizon: int
    default_quantiles: tuple[float, ...]


def _target_array(target, settings: ForecastSettings):
    """Validate one target list and return it as an FP32 NumPy array.

    One C-level pass collects element types, so booleans, strings and other
    non-numbers are rejected before NumPy could coerce them (it would accept
    ``'1.5'`` or ``True``). JSON produces only int/float; other ``Real`` types,
    such as NumPy scalars from offline callers, remain accepted as before.
    """
    import numpy as np

    if not isinstance(target, list) or not 1 <= len(target) <= settings.max_context:
        raise ValueError(f'target must contain 1 to {settings.max_context} observations')
    types = set(map(type, target))
    if not types <= {int, float} and any(issubclass(t, bool) or not issubclass(t, Real) for t in types):
        raise ValueError(_TARGET_ERROR)
    try:
        values = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):  # e.g. integers beyond float64
        raise ValueError(_TARGET_ERROR) from None
    # NaN/inf fail the comparison, so this also enforces finiteness.
    if values.shape != (len(target),) or not np.all(np.abs(values) <= _FP32_MAX):
        raise ValueError(_TARGET_ERROR)
    return values.astype(np.float32)


def parse_request(data, settings: ForecastSettings):
    """Validate a forecast request and return ``(inputs, horizon, levels)``.

    ``inputs`` are new dicts (the caller's data is not modified) whose
    ``target`` is the validated 1-D FP32 NumPy array, ready for tensor staging.
    """
    if not isinstance(data, dict) or set(data) - {'inputs', 'parameters'}:
        raise ValueError('Expected inputs and optional parameters')
    inputs = data.get('inputs')
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('inputs must be a nonempty list')
    parameters = data.get('parameters', {})
    if not isinstance(parameters, dict) or set(parameters) - {'prediction_length', 'quantile_levels'}:
        raise ValueError('Only prediction_length and quantile_levels are supported')
    horizon = parameters.get('prediction_length', settings.default_horizon)
    if type(horizon) is not int or not 1 <= horizon <= settings.max_horizon:
        raise ValueError(f'prediction_length must be an integer in [1,{settings.max_horizon}]')
    levels = parameters.get('quantile_levels', list(settings.default_quantiles))
    if (not isinstance(levels, list) or not levels
            or any(type(q) not in (int, float) or q not in settings.quantiles for q in levels)
            or levels != sorted(set(levels))):
        raise ValueError(f'quantile_levels must be increasing, unique values from {settings.quantiles}')
    parsed = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) - {'target', 'item_id'}:
            raise ValueError('Each input supports only target and item_id')
        target = _target_array(item.get('target'), settings)
        if 'item_id' in item and not isinstance(item['item_id'], str):
            raise ValueError('item_id must be a string')
        parsed.append({**item, 'target': target})
    return parsed, horizon, levels


def format_predictions(quantiles, item_ids, levels, horizon, settings: ForecastSettings):
    """Format CPU ``[rows, native quantiles, horizon]`` NumPy/Torch forecasts as JSON."""
    import numpy as np

    if tuple(quantiles.shape) != (len(item_ids), len(settings.quantiles), horizon):
        raise ValueError(f'Unexpected forecast shape: {quantiles.shape}')
    # Check only the requested quantiles, in one vectorized pass.
    selected = quantiles[:, [settings.quantiles.index(q) for q in levels]]
    if not np.isfinite(np.asarray(selected)).all():
        raise ValueError('Model returned nonfinite forecasts')
    keys = [str(q) for q in levels]
    predictions = []
    for item_id, rows in zip(item_ids, selected.tolist()):
        prediction = dict(zip(keys, rows))
        if item_id is not None:
            prediction['item_id'] = item_id
        predictions.append(prediction)
    return {'predictions': predictions}
