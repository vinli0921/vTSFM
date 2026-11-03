"""Univariate quantile IO helpers, with settings supplied by each adapter."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class ForecastSettings:
    max_context: int
    max_horizon: int
    quantiles: tuple[float, ...]
    default_horizon: int
    default_quantiles: tuple[float, ...]


def parse_request(data, settings: ForecastSettings):
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
    for item in inputs:
        if not isinstance(item, dict) or set(item) - {'target', 'item_id'}:
            raise ValueError('Each input supports only target and item_id')
        target = item.get('target')
        if not isinstance(target, list) or not 1 <= len(target) <= settings.max_context:
            raise ValueError(f'target must contain 1 to {settings.max_context} observations')
        if any(isinstance(x, bool) or not isinstance(x, Real)
               or not math.isfinite(x) or abs(x) > 3.402823466e38 for x in target):
            raise ValueError('target must be a finite univariate FP32 series')
        if 'item_id' in item and not isinstance(item['item_id'], str):
            raise ValueError('item_id must be a string')
    return inputs, horizon, levels


def format_predictions(quantiles, item_ids, levels, horizon, settings: ForecastSettings):
    if tuple(quantiles.shape) != (len(item_ids), len(settings.quantiles), horizon):
        raise ValueError(f'Unexpected forecast shape: {quantiles.shape}')
    predictions = []
    for item_id, row in zip(item_ids, quantiles):
        prediction = {str(q): row[settings.quantiles.index(q)].tolist() for q in levels}
        if not all(math.isfinite(x) for values in prediction.values() for x in values):
            raise ValueError('Model returned nonfinite forecasts')
        if item_id is not None:
            prediction['item_id'] = item_id
        predictions.append(prediction)
    return {'predictions': predictions}
