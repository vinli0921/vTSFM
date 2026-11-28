"""CPU-safe bounds for numerical requests and whole-forecast execution.

Work units are adapter-defined proxies. The resident byte calculation covers
transport/result tensors retained by the shared worker, not model weights,
attention caches, allocator reservation, or temporary model scratch. Those need
separate GPU measurements for the declared shape envelope.
"""

from dataclasses import dataclass, fields, replace
import json
from importlib import import_module
from typing import Iterable

from vtsfm.compilation import EXECUTION_ENV
from vtsfm.precision import PRECISION_ENV, TF32_OVERRIDE_ENV
from vtsfm.stepping import CONTINUOUS_ENV

FORECAST_LIMITS_ENV = 'VTSFM_FORECAST_LIMITS'


def resolve_cost(spec):
    """Load an adapter's CPU-only shape cost from its lazy descriptor."""
    path = spec.admission_cost
    if spec.admission_limits is None or not isinstance(path, str):
        raise ValueError(f'Adapter {spec.name!r} has no numerical admission policy')
    module, separator, name = path.partition(':')
    if not separator or not module or not name or ':' in name:
        raise ValueError(f'Invalid admission cost path for {spec.name!r}')
    cost = getattr(import_module(module), name)
    if not callable(cost):
        raise ValueError(f'Admission cost for {spec.name!r} must be callable')
    return cost


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True, slots=True)
class ForecastCost:
    """An adapter's conservative cost for its padded numerical request."""

    rows: int
    context: int
    horizon: int
    work_units: int
    input_bytes: int
    output_bytes: int

    def __post_init__(self):
        for field in fields(self):
            _positive_integer(getattr(self, field.name), field.name)


def forward_resident_bytes(costs: Iterable[ForecastCost]) -> int:
    """Bound shared FP32 staging/results for all slots in one worker forward.

    Count two copies of inputs, three of actual outputs, the rectangular packed
    output, and two metadata/header allowances per request. Mixed row counts and
    horizons are padded to their independent maxima. Native quantile count is
    inferred from FP32 output bytes; it is normally identical across one model's
    requests, but using the maximum remains conservative for mixed metadata.
    """
    # Resolve the shared result constant after admission types are defined:
    # adapter discovery also imports ForecastLimits into its descriptors.
    from .adapters.result_transport import MAX_METADATA_BYTES

    costs = tuple(costs)
    if not costs:
        return 0
    quantiles = []
    for cost in costs:
        if not isinstance(cost, ForecastCost):
            raise ValueError('Resident cost calculation requires ForecastCost values')
        count, remainder = divmod(cost.output_bytes, cost.rows * cost.horizon * 4)
        if not count or remainder:
            raise ValueError('Output byte count must match FP32 [rows, quantiles, horizon]')
        quantiles.append(count)
    packed = (len(costs) * max(cost.rows for cost in costs) * max(quantiles)
              * max(cost.horizon for cost in costs) * 4)
    # The result format has a 28-byte header and at most three alignment bytes.
    metadata = 2 * len(costs) * (MAX_METADATA_BYTES + 32)
    return (2 * sum(cost.input_bytes for cost in costs)
            + 3 * sum(cost.output_bytes for cost in costs) + packed + metadata)


@dataclass(frozen=True, slots=True)
class ForecastLimits:
    """Independent request, model-call, and scheduled-forward ceilings."""

    max_request_rows: int
    max_context: int
    max_horizon: int
    max_execution_rows: int
    max_execution_work: int
    max_scheduled_requests: int
    max_forward_bytes: int

    def __post_init__(self):
        for field in fields(self):
            _positive_integer(getattr(self, field.name), field.name)

    def validate_request(self, cost: ForecastCost) -> None:
        """Reject before submission, including requests too big to run alone."""
        if not isinstance(cost, ForecastCost):
            raise ValueError('Request admission requires a ForecastCost')
        checks = (
            ('request rows', cost.rows, self.max_request_rows),
            ('context', cost.context, self.max_context),
            ('horizon', cost.horizon, self.max_horizon),
            ('execution rows', cost.rows, self.max_execution_rows),
            ('execution work', cost.work_units, self.max_execution_work),
        )
        for name, actual, maximum in checks:
            if actual > maximum:
                raise ValueError(f'Forecast {name} {actual} exceeds limit {maximum}')

    def execution_row_cap(self, work_per_row: int) -> int:
        """Rows that fit one compatible call; zero means no row fits."""
        _positive_integer(work_per_row, 'work_per_row')
        return min(self.max_execution_rows, self.max_execution_work // work_per_row)

    def validate_launch(self, max_num_seqs: int, worst_request_cost: ForecastCost) -> None:
        """Bound any scheduled mixture using the caller's worst shape corner.

        This is deliberately a startup check. Throwing an admission error after
        vLLM schedules individually valid requests can terminate its V1 worker.
        The supplied corner can conservatively exceed a tighter work budget;
        actual requests still pass ``validate_request`` independently.
        """
        _positive_integer(max_num_seqs, 'max_num_seqs')
        if max_num_seqs > self.max_scheduled_requests:
            raise ValueError(f'Forecast scheduled requests {max_num_seqs} exceeds limit '
                             f'{self.max_scheduled_requests}')
        resident = forward_resident_bytes([worst_request_cost] * max_num_seqs)
        if resident > self.max_forward_bytes:
            raise ValueError(f'Forecast resident bytes {resident} exceeds limit '
                             f'{self.max_forward_bytes}; reduce rows, context, horizon or slots')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate forecast limit: {key}')
        result[key] = value
    return result


def resolve_limits(defaults: ForecastLimits, encoded_json: str | None, *,
                   ceiling: ForecastLimits | None = None,
                   inactive_eager_fp32_ceiling: ForecastLimits | None = None) -> ForecastLimits:
    """Merge partial overrides within the selected validated ceiling."""
    if encoded_json is None:
        return defaults
    if not isinstance(encoded_json, str):
        raise ValueError('Forecast limits must be an encoded JSON object')
    try:
        overrides = json.loads(encoded_json, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'Invalid forecast limits JSON: {exc}') from exc
    if not isinstance(overrides, dict):
        raise ValueError('Forecast limits must be a JSON object')
    names = {field.name for field in fields(defaults)}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(f'Unknown forecast limits: {", ".join(sorted(unknown))}')
    bounds = defaults if ceiling is None else ceiling
    for name, value in overrides.items():
        _positive_integer(value, name)
        maximum = getattr(bounds, name)
        if value > maximum:
            message = f'{name} {value} exceeds validated ceiling {maximum}'
            if (inactive_eager_fp32_ceiling is not None
                    and value <= getattr(inactive_eager_fp32_ceiling, name)):
                message += ('. Wider limits require eager FP32 whole forecasts with TF32 disabled. '
                            'Use --forecast-precision fp32 --forecast-execution eager.')
            raise ValueError(message)
    return replace(defaults, **overrides)


def resolve_adapter_limits(spec, environment) -> ForecastLimits | None:
    """Resolve the same opt-in policy in the launcher, frontend and worker."""
    if spec is None or spec.admission_limits is None:
        return None
    ceiling = None
    if (environment.get(PRECISION_ENV) == 'fp32'
            and environment.get(EXECUTION_ENV, 'eager') == 'eager'
            and environment.get(TF32_OVERRIDE_ENV, '0') == '0'
            and environment.get(CONTINUOUS_ENV) != '1'):
        ceiling = spec.eager_fp32_admission_ceiling
    inactive_ceiling = spec.eager_fp32_admission_ceiling if ceiling is None else None
    return resolve_limits(spec.admission_limits, environment.get(FORECAST_LIMITS_ENV),
                          ceiling=ceiling, inactive_eager_fp32_ceiling=inactive_ceiling)
