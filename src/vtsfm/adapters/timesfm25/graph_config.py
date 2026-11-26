"""CPU-safe settings for the experimental fixed-shape whole-forecast path."""

FORECAST_MODE_ENV = 'VTSFM_TIMESFM_FORECAST_MODE'
FORECAST_SHAPE_ENV = 'VTSFM_TIMESFM_FORECAST_SHAPE'
FORECAST_MODES = ('dense-graph', 'paged-graph', 'dense-eager', 'paged-eager')
# Rows, context length, horizon; the measured numerical envelope is deliberately narrow.
FORECAST_GRAPH_SHAPE = (2, 512, 1024)
CHILLER_GRAPH_SHAPE = (5, 512, 96)


def resolve_fixed_forecast_shape(mode, value=None):
    if value is None:
        return FORECAST_GRAPH_SHAPE
    if mode is None:
        raise ValueError('--timesfm-forecast-shape requires --timesfm-forecast-mode')
    if isinstance(value, str):
        try:
            value = tuple(int(part) for part in value.split(','))
        except ValueError as error:
            raise ValueError(f'{FORECAST_SHAPE_ENV} must be ROWS,CONTEXT,HORIZON') from error
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(size) is not int for size in value)
            or tuple(value) not in (FORECAST_GRAPH_SHAPE, CHILLER_GRAPH_SHAPE)):
        raise ValueError('Supported TimesFM forecast shapes are 2 512 1024 and 5 512 96')
    shape = tuple(value)
    if shape == CHILLER_GRAPH_SHAPE and not mode.startswith('dense-'):
        raise ValueError('TimesFM forecast shape 5 512 96 requires dense mode')
    return shape


def validate_fixed_forecast_execution(precision, execution):
    if precision not in (None, 'fp32') or execution != 'eager':
        raise ValueError('TimesFM fixed forecast modes require eager FP32 '
                         '(--forecast-execution eager --forecast-precision fp32)')
