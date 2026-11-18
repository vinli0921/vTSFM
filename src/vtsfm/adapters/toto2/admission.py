"""CPU-safe Toto work accounting for the pinned 32/768-point decoder."""

from vtsfm.admission import ForecastCost, ForecastLimits


# FP32 shape ceilings qualified on the reference A100; see numerical-admission.md.
# Model scratch/cache peaks remain separate from shared resident accounting.
DEFAULT_LIMITS = ForecastLimits(
    max_request_rows=16, max_context=8192, max_horizon=1024,
    max_execution_rows=16, max_execution_work=1401856,
    max_scheduled_requests=4, max_forward_bytes=32 * 1024 * 1024,
)


def forecast_cost(rows: int, context: int, horizon: int) -> ForecastCost:
    """Count temporal attention query/key pairs over every native block call.

    With P context patches and N forecast patches, the first call attends over
    P + min(N, 24) positions. Later calls embed the previous 24 median patches
    plus the new block and attend to retained context/median keys. Padded rows
    pay for their full width. This proxy is not FLOPs or a scratch-memory bound.

    Input bytes include left padding to 32 points. Output bytes are the compact
    public [rows, 9, horizon] result, regardless of the requested quantile subset.
    Native buffers still round the horizon up to 32 points; multi-block decoding
    keeps a cache of P + 2*N positions, including across later calls. These and
    scaler/attention/head temporaries belong to measured model scratch, not the
    shared transport/result bound.
    """
    for name, value in (('rows', rows), ('context', context), ('horizon', horizon)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    patches = (context + 31) // 32
    prediction_patches = (horizon + 31) // 32
    block_patches = min(24, prediction_patches)
    work = (patches + block_patches) ** 2
    predicted = block_patches
    while predicted < prediction_patches:
        block = min(block_patches, prediction_patches - predicted)
        work += (block_patches + block) * (patches + predicted + block)
        predicted += block
    return ForecastCost(
        rows=rows, context=context, horizon=horizon,
        work_units=rows * work,
        input_bytes=rows * patches * 32 * 4,
        output_bytes=rows * 9 * horizon * 4,
    )
