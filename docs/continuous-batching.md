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

Prefill calls can share a padded context width. AR calls require the same
physical cache position: upstream attention writes batched caches using the
first row's `next_index`. Requests at different positions use separate calls
within an iteration. A later arrival can enter while an older forecast is
active, but does not necessarily merge into that older cohort. Different initial
contexts can reach the same position; parity tests cover that case too.

Shared calls concatenate cache/state rows and copy them back into independent
request storage. Retiring a request therefore cannot retain its peers' backing
cache allocation. Completion packs the final response and releases state.
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
rows and 2,580,992 work units**, not four independent 16-row caches. New prefills
wait for capacity. Mixed-horizon calls charge the largest padded cache capacity
for every row and split within the model-call budget.

For the pinned 20-layer, 16-head, width-80 FP32 model:

```text
cache_capacity = ceil(context/32) + 4 * floor((horizon-1)/128)
persistent_KV_bytes = sum(rows * cache_capacity * 409600)
```

The factor includes K/V and both flip streams. The default full corner is
**1,861,222,400 bytes (1.73 GiB)** of persistent KV. Weights, prefill heads,
cache packing/scattering and allocator reservation need additional memory.
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
```

The parity tolerance is `rtol=atol=1e-4`. Tests cover patch/block boundaries,
ragged histories, cohort changes, reordered rows, different initial contexts at
matching cache positions, independent completion, errors and cancellation.
Bounded test-only gates prove that later requests start before older ones finish
and actual AR calls batch multiple active requests. Serving has no gate files
or sleeps. The gates are not used for performance measurements.

The maximum-shape A100 probe ran four four-row C8192/H1024 requests together,
then four 16-row requests that had to execute one at a time under the global
capacity limit. Persistent KV peaked at 1.73 GiB and returned to zero. The latter
probe peaked at **4.34 GiB allocated, 4.82 GiB reserved and 5.84 GiB sampled
device memory**. These finite synthetic probes establish measured capacity on
the pinned idle A100; they are not exact memory guarantees on other stacks or
under external GPU contention.

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
small differences. Cache concatenation/scattering and repeated runner iterations
add costs; block scheduling alone is neither a guaranteed speedup nor a novel
research claim. Evaluate its measured tradeoffs against tuned whole batching.

The [November 7 measured baseline](baselines/2026-11-07-continuous/README.md)
records the first complete 900-request comparison. Continuous mode is slower
than whole-forecast batching under homogeneous saturation; the report separates
that measured regression from the functional scheduling milestone.
