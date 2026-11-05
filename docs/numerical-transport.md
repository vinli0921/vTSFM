# Numerical request transport

TimesFM, Toto, and Moirai carry whole-forecast inputs through vLLM's raw
multimodal interface. This transport targets **vLLM 0.16.0** and uses plugin
processor hooks and built-in field reducers; it requires no vLLM source patch.
Chronos-2 keeps its separate upstream integration.

## Why batched fields matter

The earlier processor declared `context` and `horizon` as shared fields. vLLM
groups consecutive multimodal items by shared-field values. Distinct forecasts
can therefore form separate groups, while `GPUModelRunner._extract_mm_kwargs`
combines their dictionaries with `update`: later groups overwrite earlier keys.
Raising `max_num_seqs` alone does not preserve all numerical payloads.

The processor now declares two **batched** fields with a consistent field set:

| Field | One request | Worker batch |
| --- | --- | --- |
| `context` | FP32 `[series, longest_history]`, left NaN padding | `[requests, series, context]` when shapes match; otherwise a list of 2-D tensors |
| `request_metadata` | CPU uint8 vector containing UTF-8 JSON | Stacked byte vectors when lengths match; otherwise a list of byte vectors |

Before parsing, each field has a singleton outer container (`context=[tensor]`,
`request_metadata=[bytes_tensor]`), so the parser sees one multimodal item per
forecast. The processor extracts that item; vLLM subsequently batches items from
different requests and copies contexts to the worker device. Metadata uses
`keep_on_cpu=True` and is decoded without copying the numerical context back to
the CPU.

Version-1 metadata contains the request ID, item IDs, true per-series history
lengths, horizon, and requested quantiles. `decode_requests` validates the wire
layout, matching row counts/lengths, and unique request IDs. It retains different
shapes and settings without treating them as execution-compatible. Transport
order is not a guarantee about vLLM's pooled-output slots.

## Current execution boundary

Normal serving still uses **`max_num_seqs=1`**. `ForecastModel.forward` explicitly
rejects multiple transported requests until [issue #7](https://github.com/vinli0921/vTSFM/issues/7)
provides correct pooled-output routing. This change does not combine forecasts
into one model execution, implement continuous batching, or enable vLLM-managed
attention/KV caching. It makes no performance claim.

Wire metadata is separate from the [prepared execution contract](execution-contract.md).
Model adapters will prepare `PreparedRequest` tensors and masks in issues
#8/#9/#10; shared execution will consume `PreparedBatch` in #11. Normalization,
patching, model attention, and output semantics remain adapter responsibilities.

When no request ID is supplied, preprocessing generates a transport UUID. This
does not fix the existing `pending[None]` state or provide offline-call lifecycle
support; that remains [issue #13](https://github.com/vinli0921/vTSFM/issues/13).

## Manual transport probe

Run on an idle Linux GPU using a configured adapter environment:

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TRANSPORT_ENGINE=1 PYTHONPATH=src:tests \
  .venv-timesfm25/bin/python -m pytest tests/test_transport_engine.py -q -s
```

The test uses a real spawned vLLM engine with dummy weights and no checkpoint
download. A test-only worker extension imports/registers a fake architecture;
normal plugin discovery never registers it. Test-only pinned engine utility calls
pause scheduling, enqueue distinct one-placeholder requests through real IPC,
then resume and drain outputs. This makes the same-cycle check deterministic;
there is no scheduler patch or timing-based sleep. The fake model returns fingerprints of all
values and metadata, plus a worker invocation counter and received request
count. These assertions require requests to share a worker forward and survive
subsequent request-slot reuse. Copying values to the CPU for fingerprinting is
test instrumentation, not the forecast execution path.

A separate check reproduces the old shared-field overwrite through the pinned
worker extraction function. CPU metadata tests run without importing Torch or
vLLM:

```bash
.venv/bin/python -m pytest tests/test_transport.py -q
```

The engine probe is manual and separate from the bounded Modal contract suite.
Its synthetic fingerprints validate transport, not real-model numerical parity,
HTTP output routing, forecast accuracy, or throughput.
