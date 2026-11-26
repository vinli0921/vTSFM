"""CPU-safe opt-in settings for Toto's fixed-shape CUDA graph."""

from .admission import forecast_cost

GRAPH_SHAPE_ENV = 'VTSFM_TOTO_FORECAST_GRAPH_SHAPE'
FAST_SCALING_ENV = 'VTSFM_TOTO_FAST_UNIT_SCALING'
GRAPH_SCALING_MODE = 'stride-aware'


def parse_graph_shape(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = tuple(int(part) for part in value.split(','))
        except ValueError as error:
            raise ValueError(f'{GRAPH_SHAPE_ENV} must be ROWS,CONTEXT,HORIZON') from error
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(dim) is not int or dim < 1 for dim in value)):
        raise ValueError('Toto graph shape must contain three positive integers (ROWS CONTEXT HORIZON)')
    return tuple(value)


def validate_graph_shape(shape, limits):
    rows, width, horizon = shape
    if rows > limits.max_request_rows or width > limits.max_context or horizon > limits.max_horizon:
        raise ValueError('Toto graph shape exceeds the numerical admission limits')
    limits.validate_request(forecast_cost(rows, width, horizon))


def validate_graph_execution(precision, execution):
    if precision not in (None, 'fp32') or execution != 'eager':
        raise ValueError('Toto CUDA graph replay requires eager FP32 forecasts')
