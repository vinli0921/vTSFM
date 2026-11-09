"""Pinned FP32 TimesFM shape ceilings and conservative numerical work units."""

from vtsfm.admission import ForecastCost, ForecastLimits


# These ceilings define the shape envelope; supported hardware, measured scratch
# memory and safety margin are recorded separately from this metadata-only policy.
DEFAULT_LIMITS = ForecastLimits(
    max_request_rows=16, max_context=8192, max_horizon=1024,
    max_execution_rows=16, max_execution_work=2580992,
    max_scheduled_requests=4, max_forward_bytes=32 * 1024 * 1024,
)


def forecast_cost(rows: int, context: int, horizon: int) -> ForecastCost:
    """Cost the full padded width, not the sum of ragged observed lengths.

    The pinned decoder uses 32-point input patches and a 128-point output block.
    Each later block feeds back four input patches. Two decode passes implement
    flip invariance. Squared final cache length is a conservative attention-work
    proxy; it is not an exact FLOP or scratch-memory estimate. All nine native
    quantiles are computed even when the caller requests fewer output levels.
    """
    for name, value in (('rows', rows), ('context', context), ('horizon', horizon)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    patches = (context + 31) // 32
    decode_steps = (horizon - 1) // 128
    tokens = patches + 4 * decode_steps
    return ForecastCost(
        rows=rows, context=context, horizon=horizon,
        work_units=2 * rows * tokens * tokens,
        input_bytes=rows * patches * 32 * 4,
        output_bytes=rows * 9 * horizon * 4,
    )
