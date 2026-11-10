"""CPU-safe padded recursion accounting for pinned Moirai 2.0 univariate IO."""

from vtsfm.admission import ForecastCost, ForecastLimits


# Checkpoint config at 30f43ff08c8494f4943ae1521e9d4e94a0fbb389.
PATCH_SIZE = 16
OUTPUT_BLOCK_SIZE = PATCH_SIZE * 4
NATIVE_QUANTILES = 9

# Candidate FP32 envelope. Actual model/scaler/attention scratch needs separate
# GPU measurement before these ceilings are described as validated capacity.
# Eight logical rows keep the seven-sensor accuracy/integration tasks supported;
# a recursive module call expands these to 72 trajectory rows, not eight.
DEFAULT_LIMITS = ForecastLimits(
    max_request_rows=8, max_context=7168, max_horizon=1024,
    max_execution_rows=8, max_execution_work=285834752,
    max_scheduled_requests=4, max_forward_bytes=32 * 1024 * 1024,
)


def forecast_cost(rows: int, context: int, horizon: int) -> ForecastCost:
    """Charge full-sequence attention and recursive nine-quantile trajectories.

    The pinned ``Moirai2Forecast.forward`` allocates all past/future tokens once
    and runs the module on that full sequence every time. The first pass uses
    ``rows`` samples; each later 64-point block runs nine expanded trajectories
    per sample and reduces their 9x9 candidate quantiles. There is no KV reuse
    or median-only recursion in this policy.

    Work units are weighted attention-position pairs plus recursive candidate
    values, not exact FLOPs or a memory estimate. Packed scaling and attention
    masks/biases also have quadratic scratch; backend fusion is not assumed.
    Ragged rows pay for the request's longest padded history even when masked.
    The input byte count covers patch-rounded histories, while output bytes
    describe only the compact public FP32 [rows, 9, requested horizon] result.
    Expanded targets, 81-candidate reductions, full-sequence projection buffers
    and patch-rounded future predictions belong to measured model scratch.
    """
    for name, value in (('rows', rows), ('context', context), ('horizon', horizon)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    context_patches = (context + PATCH_SIZE - 1) // PATCH_SIZE
    prediction_patches = (horizon + PATCH_SIZE - 1) // PATCH_SIZE
    blocks = (prediction_patches + 3) // 4
    tokens = context_patches + prediction_patches
    module_rows = rows * (1 + NATIVE_QUANTILES * (blocks - 1))
    recursive_points = max(0, prediction_patches - 4) * PATCH_SIZE
    reduction_candidates = rows * NATIVE_QUANTILES ** 2 * recursive_points
    return ForecastCost(
        rows=rows, context=context, horizon=horizon,
        work_units=module_rows * tokens ** 2 + reduction_candidates,
        input_bytes=rows * context_patches * PATCH_SIZE * 4,
        output_bytes=rows * NATIVE_QUANTILES * horizon * 4,
    )
