"""CPU-safe contracts for prepared numerical work; no serving integration yet.

Tensor objects may be Torch tensors or CPU arrays implementing TensorLike. This
module never imports a backend, moves tensors, normalizes data, or runs a model.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence

from vtsfm.batching import BatchKey, BatchPlan, BatchRequest, _integer, _text


class TensorLike(Protocol):
    """Minimal array interface; Torch is imported only by concrete adapters."""

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def dtype(self) -> object: ...

    def tolist(self) -> list: ...

    def __getitem__(self, index: slice) -> 'TensorLike': ...


def _dtype(tensor: TensorLike) -> str:
    return str(tensor.dtype).removeprefix('torch.')


def _device(tensor: TensorLike) -> str:
    # NumPy 1.x lacks .device; NumPy 2.x and Torch both expose it.
    return str(getattr(tensor, 'device', 'cpu'))


@dataclass(frozen=True, slots=True)
class Workload:
    """Logical dimensions, NOT an estimate of latency or peak memory.

    input_patches counts all rows; forecast_blocks counts iterations per row.
    Adapter-specific recursion/flip passes and temporary storage need profiling.
    """

    rows: int
    context_points: int
    forecast_values: int
    input_patches: int | None
    forecast_blocks: int | None


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """Conservative requirements for future shared execution, not HTTP support.

    Unknown checkpoint-derived patch/block sizes stay None until runtime loading.
    execution_options records the exact model policy used by a compatible key.
    """

    max_context: int
    max_horizon: int
    quantiles: tuple[float, ...]
    supported_dtypes: tuple[str, ...] = ('float32',)
    execution_options: tuple[tuple[str, str], ...] = ()
    input_patch_size: int | None = None
    output_block_size: int | None = None
    supports_ragged_batching: bool = False
    supports_grouped_rows: bool = False
    supports_point_forecast: bool = True
    supports_block_stepping: bool = False

    def __post_init__(self):
        _integer(self.max_context, 'max_context')
        _integer(self.max_horizon, 'max_horizon')
        # Reuse compatibility-key validation/canonicalization instead of a
        # second implementation of quantile/option semantics.
        key = BatchKey('capabilities', 'metadata', 'metadata', 1, 1,
                       self.quantiles, execution_options=self.execution_options)
        object.__setattr__(self, 'quantiles', key.quantile_levels)
        object.__setattr__(self, 'execution_options', key.execution_options)
        dtypes = tuple(self.supported_dtypes)
        if not dtypes:
            raise ValueError('supported_dtypes must not be empty')
        for dtype in dtypes:
            _text(dtype, 'dtype')
        object.__setattr__(self, 'supported_dtypes', dtypes)
        for name in ('input_patch_size', 'output_block_size'):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name)
        for name in ('supports_ragged_batching', 'supports_grouped_rows',
                     'supports_point_forecast', 'supports_block_stepping'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f'{name} must be a bool')

    def validate_key(self, key: BatchKey) -> None:
        if not isinstance(key, BatchKey):
            raise TypeError('key must be a BatchKey')
        if key.context_length > self.max_context or key.prediction_length > self.max_horizon:
            raise ValueError('context or horizon exceeds adapter limits')
        if key.dtype not in self.supported_dtypes:
            raise ValueError(f'Unsupported execution dtype: {key.dtype}')
        if (not key.quantile_levels and not self.supports_point_forecast
                or any(q not in self.quantiles for q in key.quantile_levels)):
            raise ValueError('Unsupported output quantiles')
        if key.execution_options != self.execution_options:
            raise ValueError('Unsupported execution options')

    def workload(self, metadata: BatchRequest) -> Workload:
        self.validate_key(metadata.key)
        rows = sum(metadata.group_sizes)
        key = metadata.key
        patches = (rows * ((key.context_length + self.input_patch_size - 1)
                           // self.input_patch_size)) if self.input_patch_size else None
        blocks = ((key.prediction_length + self.output_block_size - 1)
                  // self.output_block_size) if self.output_block_size else None
        return Workload(rows, rows * key.context_length,
                        rows * max(1, len(key.quantile_levels)) * key.prediction_length,
                        patches, blocks)


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """Row-aligned inputs, before model-specific patching/normalization.

    context and observed_mask are [rows, padded context]. context_lengths gives
    each right-aligned history span (including any missing observations), not its
    observed count. Mask False covers left padding and missing observations;
    adapters must mask/replace those context values before numerical operations.

    Inputs must already be on the execution/model device before constructing
    this object. Validate once during preparation: observed_mask.tolist()
    synchronizes a GPU mask. Never revalidate in a decode loop. Frozen metadata
    does not make the underlying tensors immutable;
    their owner must not mutate them while execution is outstanding.
    """

    metadata: BatchRequest
    context: TensorLike
    context_lengths: tuple[int, ...]
    observed_mask: TensorLike
    item_ids: tuple[str | None, ...]
    capabilities: ExecutionCapabilities

    def __post_init__(self):
        if not isinstance(self.metadata, BatchRequest):
            raise TypeError('metadata must be a BatchRequest')
        if not isinstance(self.capabilities, ExecutionCapabilities):
            raise TypeError('capabilities must be ExecutionCapabilities')
        self.capabilities.validate_key(self.metadata.key)
        rows = sum(self.metadata.group_sizes)
        width = self.metadata.key.context_length
        if tuple(self.context.shape) != (rows, width):
            raise ValueError('context shape must be [prepared rows, context_length]')
        if _dtype(self.context) != self.metadata.key.dtype:
            raise ValueError('context dtype does not match the compatibility key')
        if tuple(self.observed_mask.shape) != (rows, width) or _dtype(self.observed_mask) != 'bool':
            raise ValueError('observed_mask must be bool with the context shape')
        if _device(self.context) != _device(self.observed_mask):
            raise ValueError('context and mask must share a device')
        lengths = tuple(self.context_lengths)
        items = tuple(self.item_ids)
        if len(lengths) != rows or len(items) != rows:
            raise ValueError('lengths and item IDs must align with prepared rows')
        for length, mask in zip(lengths, self.observed_mask.tolist()):
            _integer(length, 'context length')
            if length > width or any(mask[:width - length]) or not any(mask):
                raise ValueError('mask does not match the right-aligned history length')
        if any(item is not None and not isinstance(item, str) for item in items):
            raise TypeError('item IDs must be strings or None')
        if not self.capabilities.supports_grouped_rows and any(n != 1 for n in self.metadata.group_sizes):
            raise ValueError('adapter requires independent singleton row groups')
        if not self.capabilities.supports_ragged_batching and len(set(lengths)) > 1:
            raise ValueError('adapter has not validated ragged batching')
        object.__setattr__(self, 'context_lengths', lengths)
        object.__setattr__(self, 'item_ids', items)

    @property
    def workload(self) -> Workload:
        return self.capabilities.workload(self.metadata)


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """Bind an existing metadata plan to its ordered, uncollated inputs.

    Adapter execution owns tensor collation; constructing this does not pack
    tensors or schedule live work. Each request occurs once, in planned order.
    """

    plan: BatchPlan
    requests: tuple[PreparedRequest, ...]

    def __post_init__(self):
        if not isinstance(self.plan, BatchPlan):
            raise TypeError('plan must be a BatchPlan')
        requests = tuple(self.requests)
        if len(requests) != len(self.plan.request_slices):
            raise ValueError('requests must match every planned request slice')
        lengths = set()
        for request, part in zip(requests, self.plan.request_slices):
            if not isinstance(request, PreparedRequest):
                raise TypeError('requests must contain PreparedRequest objects')
            if request.metadata.key != self.plan.key:
                raise ValueError('request key does not match the batch key')
            if (request.metadata.request_id != part.request_id
                    or sum(request.metadata.group_sizes) != part.stop - part.start):
                raise ValueError('request identity/rows do not match the plan')
            if request.capabilities != requests[0].capabilities:
                raise ValueError('requests must use the same adapter capabilities')
            if _device(request.context) != _device(requests[0].context):
                raise ValueError('batch inputs must share a device')
            # The planner owns group-ID assignment; verify group boundaries too.
            offset = part.start
            next_group = self.plan.group_ids[offset]
            for count in request.metadata.group_sizes:
                group = self.plan.group_ids[offset:offset + count]
                if group != (next_group,) * count:
                    raise ValueError('request groups do not match the plan')
                offset += count
                next_group += 1
            lengths.update(request.context_lengths)
        if not requests[0].capabilities.supports_ragged_batching and len(lengths) > 1:
            raise ValueError('adapter has not validated ragged batching')
        object.__setattr__(self, 'requests', requests)


@dataclass(frozen=True, slots=True)
class RequestForecast:
    """Per-request output view; duplicate or absent item IDs remain valid."""

    request_id: str
    item_ids: tuple[str | None, ...]
    quantile_levels: tuple[float, ...]
    values: TensorLike


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Values in exact planned row order: [rows, requested quantiles, horizon].

    Point-only output uses one channel and an empty key.quantile_levels tuple.
    Adapters select/reorder their native channels before returning this result.
    Values remain on the input/model device until response serialization.
    """

    batch: PreparedBatch
    values: TensorLike

    def __post_init__(self):
        if not isinstance(self.batch, PreparedBatch):
            raise TypeError('batch must be a PreparedBatch')
        key = self.batch.plan.key
        expected = (len(self.batch.plan.group_ids), max(1, len(key.quantile_levels)),
                    key.prediction_length)
        if tuple(self.values.shape) != expected:
            raise ValueError(f'forecast shape must be {expected}, got {self.values.shape}')
        if _dtype(self.values) != key.dtype:
            raise ValueError('forecast dtype does not match the compatibility key')
        if _device(self.values) != _device(self.batch.requests[0].context):
            raise ValueError('forecasts must remain on the input/model device')

    def split(self) -> dict[str, RequestForecast]:
        return {
            part.request_id: RequestForecast(part.request_id, request.item_ids,
                self.batch.plan.key.quantile_levels, self.values[part.start:part.stop])
            for part, request in zip(self.batch.plan.request_slices, self.batch.requests)
        }


class ForecastExecutor(Protocol):
    """Adapter boundary implemented by #8/#9/#10, consumed by the #11 worker."""

    def execute(self, batch: PreparedBatch) -> ExecutionResult: ...
