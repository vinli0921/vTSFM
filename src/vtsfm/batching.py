"""CPU-only planning metadata for prepared, row-aligned forecast requests.

Adapters own preparation and attention-group semantics, including covariates and
cross-learning. This planner neither packs tensors nor schedules live execution.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, TypeVar


def _text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class BatchKey:
    """Conservative compatibility; empty quantiles describe point-only outputs."""

    adapter: str
    model_id: str
    model_revision: str
    context_length: int
    prediction_length: int
    quantile_levels: tuple[float, ...]
    dtype: str = "float32"
    execution_options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("adapter", "model_id", "model_revision", "dtype"):
            _text(getattr(self, name), name)
        _integer(self.context_length, "context_length")
        _integer(self.prediction_length, "prediction_length")
        levels = tuple(self.quantile_levels)
        if any(isinstance(level, bool) or not isinstance(level, Real) for level in levels):
            raise TypeError("quantile_levels must contain real probabilities")
        levels = tuple(float(level) for level in levels)
        if any(not 0 < level < 1 for level in levels):
            raise ValueError("quantile_levels must be finite and between 0 and 1")
        if any(left >= right for left, right in zip(levels, levels[1:])):
            raise ValueError("quantile_levels must be strictly increasing")
        object.__setattr__(self, "quantile_levels", levels)
        options = []
        for pair in self.execution_options:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError('execution options must be (name, string value) pairs')
            name, value = pair
            _text(name, 'execution option name')
            if not isinstance(value, str):
                raise TypeError('execution option values must be strings')
            options.append((name, value))
        if len({name for name, _ in options}) != len(options):
            raise ValueError('duplicate execution option names')
        object.__setattr__(self, 'execution_options', tuple(sorted(options)))


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """Atomic prepared rows grouped by adapter-defined attention independence."""

    request_id: str
    key: BatchKey
    group_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        if not isinstance(self.key, BatchKey):
            raise TypeError("key must be a BatchKey")
        groups = tuple(self.group_sizes)
        if not groups:
            raise ValueError("group_sizes must not be empty")
        for size in groups:
            _integer(size, "group size")
        object.__setattr__(self, "group_sizes", groups)


@dataclass(frozen=True, slots=True)
class RequestSlice:
    """A request's half-open range in a plan's prepared rows."""

    request_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        _integer(self.start, "start", minimum=0)
        _integer(self.stop, "stop")
        if self.stop <= self.start:
            raise ValueError("request slice must contain at least one row")


OutputRow = TypeVar("OutputRow")


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Immutable row/group layout with contiguous, non-overlapping request slices."""

    key: BatchKey
    group_ids: tuple[int, ...]
    request_slices: tuple[RequestSlice, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, BatchKey):
            raise TypeError("key must be a BatchKey")
        groups = tuple(self.group_ids)
        slices = tuple(self.request_slices)
        if not groups or not slices:
            raise ValueError("a batch plan must contain rows and requests")
        previous_group = -1
        for group in groups:
            _integer(group, "group ID", minimum=0)
            if group not in (previous_group, previous_group + 1):
                raise ValueError("group IDs must be contiguous and start at zero")
            previous_group = group
        position = 0
        request_ids = set()
        for part in slices:
            if not isinstance(part, RequestSlice):
                raise TypeError("request_slices must contain RequestSlice metadata")
            if part.request_id in request_ids:
                raise ValueError(f"duplicate request ID: {part.request_id}")
            request_ids.add(part.request_id)
            if part.start != position or part.stop > len(groups):
                raise ValueError("request slices must cover consecutive rows")
            if position and groups[position] == groups[position - 1]:
                raise ValueError("attention groups must not span requests")
            position = part.stop
        if position != len(groups):
            raise ValueError("request slices must cover every row")
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "request_slices", slices)

    def split_outputs(self, outputs: Iterable[OutputRow]) -> dict[str, tuple[OutputRow, ...]]:
        """Route one output per prepared row; adapters filter target/covariate rows.

        The caller must preserve the planned row order. This method cannot infer
        model-specific output axes or detect a permutation of the same row count.
        """
        rows = tuple(outputs)
        if len(rows) != len(self.group_ids):
            raise ValueError(
                f"output row count {len(rows)} does not match planned rows {len(self.group_ids)}"
            )
        return {
            part.request_id: rows[part.start:part.stop] for part in self.request_slices
        }


def plan_batches(
    requests: Iterable[BatchRequest], *, max_rows: int, max_requests: int
) -> tuple[BatchPlan, ...]:
    """Plan atomic requests in FIFO order, flushing at key or budget boundaries.

    Group IDs are unique across requests within each plan. No request is split,
    and an individually oversized request fails instead of exceeding a budget.
    """
    _integer(max_rows, "max_rows")
    _integer(max_requests, "max_requests")
    plans = []
    key = None
    group_ids = []
    request_slices = []
    next_group_id = 0
    request_ids = set()

    for request in requests:
        if not isinstance(request, BatchRequest):
            raise TypeError("requests must contain BatchRequest metadata")
        if request.request_id in request_ids:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        request_ids.add(request.request_id)
        rows = sum(request.group_sizes)
        if rows > max_rows:
            raise ValueError(f"request {request.request_id!r} exceeds max_rows={max_rows}")
        if request_slices and (
            request.key != key
            or len(group_ids) + rows > max_rows
            or len(request_slices) == max_requests
        ):
            plans.append(BatchPlan(key, tuple(group_ids), tuple(request_slices)))
            group_ids = []
            request_slices = []
            next_group_id = 0
        key = request.key
        start = len(group_ids)
        for size in request.group_sizes:
            group_ids.extend([next_group_id] * size)
            next_group_id += 1
        request_slices.append(RequestSlice(request.request_id, start, len(group_ids)))

    if request_slices:
        plans.append(BatchPlan(key, tuple(group_ids), tuple(request_slices)))
    return tuple(plans)
