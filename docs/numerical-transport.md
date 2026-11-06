# Numerical request transport and output routing

TimesFM, Toto, and Moirai carry whole-forecast inputs through vLLM's raw
multimodal interface. This transport targets **vLLM 0.33.0 with Model Runner
V1** and uses plugin processor hooks and built-in field reducers; it requires no
vLLM source patch. Chronos-2 keeps its separate upstream integration on 0.16.0.
The launcher sets `VLLM_USE_V2_MODEL_RUNNER=0`; the released V2 runner excludes
the `plugin` pooling task.

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

## Output routing

Transport order can differ from worker slot order after scheduling and slot
reuse. vLLM's pooling metadata does not contain request IDs, and its internal IDs
can differ from the original numerical IDs. Routing must resolve those identities
without assuming arrival order or stripping suffixes from internal IDs.

`vtsfm.worker.ForecastWorker` subclasses the pinned worker. After loading the
model, it installs a callback that reads the current `input_batch.req_ids` and
resolves each slot's cached `mm_features` metadata to the original request ID.
The callback runs when the model executes, after worker state updates. It does
not edit vLLM source or replace a runner method. This bridge supports **vLLM
0.33.0 V1, eager execution, and one GPU**; other configurations fail explicitly.

`ForecastModel` then:

1. Matches transported requests to worker slot IDs exactly once, rejecting
   missing, duplicate, or unknown identities.
2. Runs each request's upstream forecast separately, stages its output on the
   context device, then validates `[series, native_quantiles, horizon]` and dtype.
3. Pads the validated outputs into one worker tensor in slot order, shaped
   `[requests, maximum_series_count, native_quantiles, maximum_horizon]`.

`ForecastPooler` crops each slot back to its own variable-size tensor. Pooling
consumes the shape layout once; it does not retain forecast tensors in a pending
dictionary. IO selects each request's quantiles and restores its item IDs. Wrong
row counts or output axes produce errors rather than another request's results.

## Current execution boundary

Normal serving retains **`max_num_seqs=1`**. Multiple scheduled requests are an
explicit experimental option for TimesFM, Toto, and Moirai:

```bash
./scripts/serve.sh --adapter toto2 --max-num-seqs 3 --port 8001
```

The flag changes scheduling capacity. Each request still runs an individual
whole forecast; [issue #11](https://github.com/vinli0921/vTSFM/issues/11) will
collate compatible requests into shared numerical execution. Stable numerical
workload/memory admission remains [issue #12](https://github.com/vinli0921/vTSFM/issues/12),
so the request-count flag is not a memory bound. Mixed shapes can increase the
padded output allocation. This path does not implement
continuous batching or vLLM-managed attention/KV caching and makes no accuracy
or performance claim.

Wire metadata is separate from the [prepared execution contract](execution-contract.md).
TimesFM uses its tensor forecast path for each scheduled request, and shared
execution will consume `PreparedBatch` in #11. Normalization, patching, model
attention, and output semantics remain adapter responsibilities. Moirai's
current NumPy wrapper returns CPU tensors, which the worker stages on the
execution device.

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

The transport probe remains manual and separate from the bounded Modal contract
suite. Its synthetic fingerprints validate transport; the routing suite below
checks output association.

## Manual routing probe

Run the routing suite on an idle Linux GPU in the same configured environment:

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ROUTING_ENGINE=1 PYTHONPATH=src:tests \
  .venv-timesfm25/bin/python -m pytest tests/test_routing_engine.py -q -s
```

This suite uses a deterministic fake runtime with the production worker, model,
pooler, and IO boundaries. Real-engine cases compare requests with one, two, and
three series against isolated execution, including different horizons and
quantiles, reordered submissions, and request-slot reuse. Test-only reversal of
the received payload order checks that routing follows actual worker identities.
The HTTP cases check 13 unique responses across isolated, concurrent, reordered,
and repeated requests. Malformed output-shape checks also cover recovery and
one-time pooling-layout consumption.

The fake architecture and its instrumentation are registered only by tests.
An optional real-model HTTP check compares isolated and concurrent requests in
the selected model's environment, for example:

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ROUTING_ENGINE=1 VTSFM_ROUTING_REAL_ADAPTER=toto2 \
  PYTHONPATH=src:tests .venv-toto2/bin/python -m pytest tests/test_routing_engine.py -k real_model -q -s
```

This check covers 12 requests with three different series counts, horizons, and
quantile selections. It compares the serving path to itself under concurrency;
use the separate native parity suite for an independent model reference. These
routing checks do not establish forecast accuracy, throughput, or a batching
speedup. CPU routing checks need neither Torch nor vLLM:

```bash
.venv/bin/python -m pytest tests/test_transport.py tests/test_routing.py -q
```
