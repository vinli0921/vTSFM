"""CPU-safe contract and call planning for adapter-owned forecast blocks."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, Sequence


CONTINUOUS_ENV = 'VTSFM_CONTINUOUS_BATCHING'
CONTINUOUS_TRACE_ENV = 'VTSFM_CONTINUOUS_TRACE'
PREFILL_GROUP_STEPS_ENV = 'VTSFM_PREFILL_GROUP_STEPS'


def validate_block_configuration(config):
    """Reject mismatched direct LLM launches before constructing model weights."""
    scheduler = config.scheduler_config
    parallel = config.parallel_config
    if (parallel.worker_cls != 'vtsfm.block_worker.ForecastBlockWorker'
            or scheduler.scheduler_cls != 'vtsfm.continuous_scheduler.ForecastBlockScheduler'
            or scheduler.async_scheduling or not scheduler.enable_chunked_prefill
            or config.use_v2_model_runner or not config.model_config.enforce_eager
            or config.model_config.runner_type != 'pooling'
            or config.speculative_config is not None
            or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1):
        raise ValueError('TimesFM block mode requires ForecastBlockWorker, ForecastBlockScheduler, '
                         'synchronous V1 pooling, chunked prefill and eager execution on one GPU')


class ResumableForecastAdapter(Protocol):
    """State is request-local and stays on the adapter's execution device.

    A state exposes rows, context_length, horizon, blocks_done, bucket, and
    cache_capacity. ``prefill``/``advance`` mutate compatible states once;
    ``finalize`` returns native FP32 [series, quantiles, horizon]. ``release``
    is idempotent and removes all request-owned tensor references.
    """

    def initialize(self, context: object, horizon: int) -> object: ...
    def prefill(self, states: Sequence[object]) -> None: ...
    def advance(self, states: Sequence[object]) -> None: ...
    def complete(self, state: object) -> bool: ...
    def finalize(self, state: object) -> object: ...
    def release(self, state: object) -> None: ...
    def model_call_work(self, states: Sequence[object]) -> int: ...


@dataclass(frozen=True)
class BlockRequest:
    request_id: str
    rows: int
    bucket: tuple
    work_per_row: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError('Block request ID must be nonempty')
        if any(type(value) is not int or value < 1 for value in (self.rows, self.work_per_row)):
            raise ValueError('Block rows and work must be positive integers')
        if not isinstance(self.bucket, tuple) or not self.bucket:
            raise ValueError('Block compatibility bucket must be a nonempty tuple')
        hash(self.bucket)


def plan_block_calls(requests, *, max_rows, max_work, fusion_rows):
    """Bucket independent states, preserving FIFO order within each bucket.

    A mixed-capacity call pays the maximum per-row work of any member, because
    the adapter pads its batched cache. A request above the optional fusion cap
    can execute alone if it fits the actual execution limits.
    """
    for value in (max_rows, max_work, fusion_rows):
        if type(value) is not int or value < 1:
            raise ValueError('Block execution budgets must be positive integers')
    buckets = OrderedDict()
    identities = set()
    for request in requests:
        if request.request_id in identities:
            raise ValueError('Duplicate block request ID')
        identities.add(request.request_id)
        if request.rows > max_rows or request.rows * request.work_per_row > max_work:
            raise ValueError('Block request cannot fit standalone execution')
        buckets.setdefault(request.bucket, []).append(request)
    plans = []
    for bucket in buckets.values():
        group, rows, work_per_row = [], 0, 0
        for request in bucket:
            combined_rows = rows + request.rows
            combined_work = combined_rows * max(work_per_row, request.work_per_row)
            if group and (combined_rows > min(max_rows, fusion_rows) or combined_work > max_work):
                plans.append(tuple(group))
                group, rows, work_per_row = [], 0, 0
            group.append(request.request_id)
            rows += request.rows
            work_per_row = max(work_per_row, request.work_per_row)
            if request.rows > fusion_rows:
                plans.append(tuple(group))
                group, rows, work_per_row = [], 0, 0
        if group:
            plans.append(tuple(group))
    return tuple(plans)
