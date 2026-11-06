# Prepared forecast execution contract

`vtsfm.execution` defines the boundary between shared runtime work and model
adapters. It imports no Torch, vLLM, NumPy, or model packages. Its validation and
the CPU example can be used before the live worker integration exists.

TimesFM implements `ForecastExecutor` for prepared batches of independent rows
with equal true history lengths and fully observed values after left padding.
The live serving wrapper can use this contract for compatible HTTP requests
scheduled together. Individual forecasts use the same tensor forecast core.

## Inputs, planning, and outputs

| Type | Contract |
| --- | --- |
| `PreparedRequest` | One `BatchRequest`, `context` and boolean `observed_mask` shaped `[rows, context_length]`, per-row `context_lengths` and `item_ids`, and adapter capabilities |
| `PreparedBatch` | An existing `BatchPlan` plus exactly its requests in planned order; validates IDs, row/group boundaries, keys, devices, and capabilities without collating tensors |
| `ForecastExecutor` | Adapter protocol: `execute(batch: PreparedBatch) -> ExecutionResult` |
| `ExecutionResult` | Values shaped `[total rows, requested quantiles, prediction_length]` in exact planned row order; `split()` returns `RequestForecast` objects keyed by request ID |

`context_lengths` describes the right-aligned history span, including missing
observations; it is not the number of observed values. For width 4, length 3,
and mask `[False, True, False, True]`, the first slot is padding and the third is
a missing observation inside the history. Adapters must mask or replace every
unobserved value before numerical operations. Every row needs an observation.
The prepared boundary does not normalize or patch the context.

Inputs and masks must already share the execution/model device before creating
`PreparedRequest`. Outputs retain that device and the key's dtype until response
serialization. Transfer CPU staging inputs during adapter/worker preparation;
`execute()` cannot return CUDA outputs for a batch whose inputs remain on CPU.
Validation reads `observed_mask.tolist()`, synchronizing a GPU mask once during
preparation; do not reconstruct requests in a decode loop. Frozen dataclasses
protect metadata, not tensor contents: owners must keep tensors unchanged while
work is outstanding.

Request IDs are unique. Item IDs may repeat or be absent. Each prepared row maps
to one output row, and group boundaries preserve adapter-defined independence.
An empty quantile tuple represents point forecasts with one output channel when
the adapter declares point-only support. Otherwise adapters select and order
native quantile channels before constructing the result.

## Compatibility and capabilities

Use the existing `plan_batches()` policy. `BatchKey` includes model identity and
revision, padded context, prediction length, requested quantiles, dtype, and
`execution_options`. Options are unique `(name, string value)` pairs, copied and
sorted canonically. For example, `(('decode_block_size', '768'),)` is part of
Toto's declared policy. Different options produce separate plans; unsupported
options fail capability validation.

`AdapterSpec.execution_capabilities` is pure metadata. The first declarations
are conservative requirements for future execution, not changes to current HTTP
support:

| Adapter | Context / horizon limit | Input patch / output block | Execution dtype |
| --- | --- | --- | --- |
| TimesFM 2.5 | 8192 / 1024 | 32 / 128 | FP32 |
| Toto 2 | 8192 / 1024 | checkpoint-configured (`None`) / 768 | FP32 |
| Moirai 2.0 | 7168 / 1024 | checkpoint-configured (`None`) / checkpoint-configured (`None`) | FP32 |

All three declare independent singleton groups, nine available quantiles
(0.1–0.9), no point-only output, and no validated ragged batching or block
stepping. Their existing HTTP paths still accept unequal histories; these flags
gate the new contract until adapter parity tests establish safe collation.
Unknown patch/block sizes stay `None` until the loaded checkpoint supplies them.
Chronos-2 has no declaration for this boundary yet.

TimesFM's HTTP path supports ragged histories within a request. Its outer
normalization includes the zero left padding up to the next 32-point patch of
the longest history in that request. The prepared executor keeps equal true
history lengths across requests to preserve the isolated normalization width.
It checks finite values and the observed mask before decoding, which uses a
scalar validation sync and rejects internal gaps. The HTTP path validates
finite histories at request parsing.

`PreparedRequest.workload` reports logical rows, context points, forecast values,
and patch/block counts when sizes are known. These are dimensions, not predicted
latency or memory. Flip passes, recursive quantile trajectories, temporary
storage, and runtime caches require model-specific measurement and admission
logic (#12).

## CPU example

From a checkout with its CPU environment installed, run:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import numpy as np
from examples.fake_adapter import LastValueAdapter
from vtsfm.batching import BatchKey, BatchRequest, plan_batches
from vtsfm.execution import ExecutionCapabilities, PreparedBatch, PreparedRequest

key = BatchKey('fake', 'fake/checkpoint', 'revision', 4, 2, (0.5,))
capabilities = ExecutionCapabilities(8, 8, (0.5,))

def prepare(request_id, rows):
    context = np.array(rows, dtype=np.float32)
    count = len(rows)
    return PreparedRequest(
        BatchRequest(request_id, key, (1,) * count), context,
        (4,) * count, np.ones(context.shape, dtype=bool),
        tuple(f'item-{i}' for i in range(count)), capabilities)

requests = (
    prepare('first', [[1, 2, 3, 4], [5, 6, 7, 8]]),
    prepare('second', [[9, 10, 11, 12]]),
)
plan, = plan_batches([r.metadata for r in requests], max_rows=3, max_requests=2)
result = LastValueAdapter().execute(PreparedBatch(plan, requests)).split()
assert result['first'].values[:, 0, :].tolist() == [[4, 4], [8, 8]]
assert result['second'].values[:, 0, :].tolist() == [[12, 12]]
print(plan.group_ids)  # (0, 1, 2)
PY
```

`examples/fake_adapter.py` uses NumPy and simply repeats the last observation.
It illustrates the boundary and is not a learned model or a performance baseline.
The example is checkout-only; `examples` is not an installed wheel API.

Run the associated checks with
`.venv/bin/python -m pytest tests/test_execution.py tests/test_batching.py -q`.
CPU tests exercise the contract; live serving correctness and model numerical
parity require their later integration checks.
