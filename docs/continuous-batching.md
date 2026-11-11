# TimesFM forecast-block continuous batching

The optional block path lets requests join and finish between TimesFM forecast
blocks. Each `POST /pooling` request receives its complete forecast independently;
the numerical response schema is unchanged. Normal serving still uses whole
forecasts. Partial forecast streaming is not implemented or required.

```bash
./scripts/setup.sh --adapter timesfm25
./scripts/serve.sh --adapter timesfm25 --continuous-batching --port 8001
```

Defaults: four scheduled slots, up to 16 fused rows, and 16 waiting/running
requests per API process. The launcher supports two to four slots; one client
can use a four-slot server. Optional settings are explicit:

```bash
./scripts/serve.sh --adapter timesfm25 --continuous-batching \
  --max-num-seqs 3 --forecast-batch-rows 8 --forecast-queue-limit 16 \
  --forecast-limits limits.json --dry-run
```

Remove `--dry-run` to serve. Numerical policies retain the tighter-only semantics
in [numerical admission](numerical-admission.md). The queue cap reuses vLLM's
API admission and **HTTP 503** response before EngineCore submission. It counts
waiting plus running requests per API process; the normal CLI uses one process.

## State and batching

TimesFM produces the first 128 forecast points during prefill, then feeds back
four 32-point patches per later 128-point block. The adapter resumes this original
algorithm; it does not approximate it with repeated shorter-horizon forecasts.
Each request owns its normalization, masks, positivity flag, independent normal
and flipped streams, running statistics, median feedback, quantile spread,
output buffer and KV caches. Feedback stays separate between flip streams.
Quantiles, crossing correction and de-normalization preserve pinned semantics.
Prefill keeps separate flip-pass calls. AR combines normal and flipped rows in
one neural call while retaining each stream's own statistics and feedback.

Prefill calls can share a padded context width. AR calls can also combine
requests at different physical cache positions. The pinned upstream attention
writes every cache row through the first row's index, so simply combining those
requests would corrupt state. Shared calls instead batch the native tokenizer,
projections, RoPE, normalization, MLP and forecast heads. Equal-capacity groups
temporarily gather layer K/V for one native attention call with their native
row masks; unequal capacities keep separate native attention calls. Each
request writes its four new patches at its CPU-known position, without reading
a CUDA counter back to the host. Attention keeps that owner's original cache
width. Singleton AR also combines its two flip streams. Class-source guards
require revalidation before changing the pinned implementation; model methods
and weights are not replaced.

Each request retains its own persistent cache allocations. AR calls update those
exact buffers in place; temporary per-layer attention packs do not replace them
and require no cache copyback. Allocations and existing
prefixes persist when requests have different capacities or are reordered.
Initial shared prefill splits its cache once, so retiring a request cannot
retain a peer's backing allocation.
Completion packs the final response and releases state.
Cancellation also releases state, including zero-compute housekeeping steps.
Cleanup is idempotent. Input/queue/response-formatting errors affect their own
requests; a device or model-worker failure can still require an engine restart.

## Pinned vLLM bridge

This path requires **vLLM 0.33.0 Model Runner V1, synchronous scheduling, eager
FP32, one GPU and FCFS admission**. Installed vLLM files are unchanged.

- Arrays remain raw FP32 multimodal inputs. A prompt marker represents one
  forecast block; horizon `H` uses `ceil(H/128)` markers. Markers contain neither
  observations nor text tokenizations.
- `ForecastBlockScheduler` subclasses the pinned scheduler and schedules one
  marker per request per iteration. vLLM retains ownership of queues, slots,
  progress, completion and cancellation.
- `ForecastBlockWorker` resolves actual cached engine IDs and CPU progress;
  slot compaction never relies on arrival order.
- The block pooler returns `None` until completion, then the existing result
  tensor. vLLM's pooling cursor returns finished requests independently.

Caches belong to the adapter, **not vLLM PagedAttention/KV**. vLLM considers this
wrapper attention-free because its runner does not execute the attention.
EngineCore disables chunking for no-KV wrappers; the pinned scheduler restores
it specifically for numerical block markers, using its private prefill lookahead
hook. Revalidate these isolated hooks before changing vLLM versions.

Invalid direct launches fail before model construction. Direct `LLM` users must
select both `vtsfm.block_worker.ForecastBlockWorker` and
`vtsfm.continuous_scheduler.ForecastBlockScheduler`, enable chunked prefill,
disable async scheduling and consistently set `VTSFM_CONTINUOUS_BATCHING=1`.
HTTP is the primary validated block interface; default offline calls retain
whole forecasts. Only TimesFM advertises block capability. Toto/Moirai retain
their existing whole-forecast paths.

## Budgets and fairness

Both flip-pass caches now persist between iterations. All active requests
together must fit the configured execution-row/work limits: by default **16
independent input series and 2,580,992 work units**, not four independent 16-row caches. New prefills
wait for capacity. Mixed-horizon calls conservatively charge the largest cache
capacity for every row and split within the model-call budget, even though each
request's attention uses its own width.

Admission and fusion rows count prepared independent series. Combining both
flip passes gives an internal AR neural batch of `2*sum(series_count)`, at most
32 rows. The existing work charge already includes both passes. All ceilings
and persistent KV accounting remain unchanged; the larger internal AR batch
and temporary attention packs need fresh GPU parity and peak-memory checks.
The historical capacity measurements below predate this variant.

For the pinned 20-layer, 16-head, width-80 FP32 model:

```text
cache_capacity = ceil(context/32) + 4 * floor((horizon-1)/128)
persistent_KV_bytes = sum(rows * cache_capacity * 409600)
```

The factor includes K/V and both flip streams. The default full corner is
**1,861,222,400 bytes (1.73 GiB)** of persistent KV. Weights, prefill heads,
initial shared-prefill splitting and allocator reservation need additional memory.
The old 32 MiB accounting ceiling covers shared numerical/result staging only.

If the oldest waiter cannot fit, later admissions wait for that iteration while
active requests advance. Finite horizons and that reservation prevent a stream
of small newcomers from indefinitely starving an older large request. Host
request count is bounded separately by the API queue cap.

## Checks and reproduction

Run separately on an idle Linux GPU: the first command owns a native CUDA model.

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_STEPPING=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_stepping.py -q -s
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_CONTINUOUS=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_continuous.py -q -s
CUDA_VISIBLE_DEVICES=0 VTSFM_TEST_ADAPTER=timesfm25 \
  VTSFM_TEST_OUTPUT=results/timesfm-block-integration-NEW PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_block_integration.py -q -s
```

The parity tolerance is `rtol=atol=1e-4`. Tests cover patch/block boundaries,
ragged histories, cohort changes, reordered rows, matching and different physical
cache positions, stable cache allocations, unchanged prefixes, independent
completion, errors and cancellation. Traces include one `cache_positions` value
per request; the older scalar `cache_position` is null for mixed-position calls.
Bounded test-only gates prove that later requests start before older ones finish
and actual AR calls batch multiple active requests. Serving has no gate files
or sleeps. The gates are not used for performance measurements.

The original block implementation's maximum-shape A100 probe ran four four-row
C8192/H1024 requests together, then four 16-row requests that had to execute one
at a time under the global
capacity limit. Persistent KV peaked at 1.73 GiB and returned to zero. The latter
probe peaked at **4.34 GiB allocated, 4.82 GiB reserved and 5.84 GiB sampled
device memory**. These finite synthetic probes establish measured capacity on
the pinned idle A100; they are not exact memory guarantees on other stacks or
under external GPU contention.

On 2026-11-11, the changed stepper passed all 21 pinned-checkpoint checks,
all three real HTTP checks, and six independent native CUDA versus block-mode
HTTP integration cases on the A100. The latter include pinned AssetOpsBench
sensor histories and matched native outputs exactly. A traced shared decode
call used cache positions `[12, 12, 4]`. Cancellation, postprocessing failure, independent
completion and maximum-capacity deferral returned all request state to zero.
The capacity probes retained the 1.73 GiB persistent KV bound, with a maximum
of 3.67 GiB Torch allocated and 4.02 GiB reserved across the two probes. This
was authorized shared-GPU correctness work; whole-device NVML residency included
an existing unrelated job. All owned servers exited and that job was preserved.
These observations qualify the tested capacity and lifecycle behavior;
performance comparisons still require an idle GPU and matching baselines.

Tiny official-model CPU regressions need the existing TimesFM environment and
no downloaded weights:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_block_decode.py \
  tests/test_timesfm_block_state.py -q
```

They explicitly initialize nonzero RMS scales so the mixed-position negative
control reaches meaningful attention. The original scalar-write native batch
must fail that control while independent native rows pass. These checks establish
numerical behavior and original-cache ownership, including temporary-pack cleanup.
Issue #49's acceptance still needs the manual matched whole-versus-continuous
comparison below. The standalone path executes native eager math for both
whole forecasts and shared block decoding.

## Offered-load comparison

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m benchmarks.continuous \
  --data-dir data --output results/continuous-NEW --requests 100 \
  --sparse-interval 1.0 --diagnostic-requests 24
```

Single-slot whole forecasting, tuned four-slot whole batching and block mode
use identical pins, precision, queue caps and recorded arrival traces. Sparse
arrivals rotate mixed cases at one-second intervals. Saturation uses compatible
C512/H1024 requests; bursts interleave context/horizon/series-count workloads.
AssetOpsBench MetroPT histories retain documented causal fills. Queries repeat
correlated windows; no future labels or accuracy scores are fabricated.

Primary timing runs disable scheduler/worker trace writes. Separate diagnostic
servers replay a prefix to measure **engine enqueue to first scheduled block**.
That diagnostic excludes API preprocessing/IPC time before engine enqueue and
must not be interpreted as total server queue wait. Untraced end-to-end latency
includes those costs. Diagnostic timings do not replace primary latency samples.

Reports save requests/s, series/s, p50/p95, offered-arrival sojourn/dispatch lag,
queue diagnostics, rejections, failures, Torch allocator counters, 20 ms NVML
samples, exact commands, source hashes, and partial results. Use at least 100
requests per point, then paired repeats/reversed mode orders before interpreting
small differences. Repeated runner iterations and per-request attention calls
add overhead. Evaluate the measured tradeoffs against tuned whole batching;
block scheduling alone establishes no speedup.

The [November 7 measured baseline](baselines/2026-11-07-continuous/README.md)
records the first complete 900-request comparison. Continuous mode is slower
than whole-forecast batching under homogeneous saturation; the report separates
that measured regression from the functional scheduling milestone.
