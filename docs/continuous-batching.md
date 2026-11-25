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
After each block, the stepper writes the difference between the two median
forecasts into the request's output buffer. When the request finishes, it
combines that history with the two prefill spreads to assemble all nine
quantiles, correct crossings, reverse normalization and apply positivity once.
A request canceled between blocks releases its buffers before that final work.
The [A100 quantile assembly benchmark](baselines/2026-11-20-timesfm-quantile-assembly/README.md) compares it with both whole-forecast and the previous block epilogue.
Prefill and autoregressive (AR) decode batch the normal and sign-flipped rows
in one neural call per phase. Each stream retains its own statistics and feedback.

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

During eager CUDA FP32 decode, equal-capacity groups with dense cache rows use
one Triton copy kernel per layer. It writes the four new patches into the
request-owned K/V buffers and gathers the full-width K/V for native attention.
The forward stages a pointer table once and reuses a single pair of scratch
buffers across layers. With different capacities, each group still uses native
attention over its own cache. Ordinary decode also batches the layer counter
updates per owner, with a separate slice for each layer. The fixed-shape CUDA
graph experiment selects its original implementation for both controls.
The [H1024 A100 benchmark](baselines/2026-11-20-timesfm-decode-kv/README.md)
compares HTTP throughput, cache operations and peak memory for this change.

Cache allocations and existing prefixes persist when requests are reordered.
Initial shared prefill splits its cache by request and flip stream. Owners
with at most 18,022,400 bytes of K/V per flip retain contiguous per-layer
views of a packed buffer, trimmed to their forecast capacity. Write positions
and mask counts each have their own typed buffer. Larger owners use per-layer
clones to bound GPU memory at the admission ceiling. A singleton adopts its
original cache in place.
The [A100 prefill storage benchmark](baselines/2026-11-20-timesfm-prefill-kv/README.md)
measures copy submissions and peak memory, and checks when a retiring
peer's allocations become reclaimable.
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
independent input series and 2,580,992 work units**, not four independent 16-row
caches. New prefills wait for capacity. Mixed-horizon calls charge the largest cache
capacity for every row and split within the model-call budget, even though each
request's attention uses its own width.

Admission and fusion rows count prepared independent series. Combining both
flip passes gives an internal AR neural batch of `2*sum(series_count)`, at most
32 rows. The existing work charge already includes both passes. All ceilings
and persistent KV accounting remain unchanged. The standalone eager decoder
at `b9fffd25` passed the native-parity and capacity checks described below,
including the 32-row internal AR capacity corner.

For the pinned 20-layer, 16-head, width-80 FP32 model:

```text
cache_capacity = ceil(context/32) + 4 * floor((horizon-1)/128)
persistent_KV_bytes = sum(rows * cache_capacity * 409600)
```

The factor includes K/V and both flip streams. The default full corner is
**1,861,222,400 bytes (1.73 GiB)** of persistent KV. Weights, prefill heads,
temporary attention gathers, initial shared-prefill splitting and allocator
reservation need additional memory.
The old 32 MiB accounting ceiling covers shared numerical/result staging only.

If the oldest waiter cannot fit, later admissions wait for that iteration while
active requests advance. Finite horizons and that reservation prevent a stream
of small newcomers from indefinitely starving an older large request. Host
request count is bounded separately by the API queue cap.

## Checks and reproduction

### Experimental prefill grouping

Set `VTSFM_PREFILL_GROUP_STEPS=1` when starting a TimesFM continuous server
to let an already running decode finish a block before admitting the oldest
waiting request. The scheduler considers a hold only when this can form a
larger group of compatible queued prefills within the slot, row and work
limits. With four configured slots it examines at most the first four
requests in the FIFO queue. A one-block forecast in the
examined prefix bypasses the hold. An idle worker starts immediately.

The default is `0`. A value of `2` allows up to two scheduler iterations.
The deadline belongs to the oldest waiter and cannot renew while it stays at
the head. The bound applies to the extra grouping hold. Normal waits for
busy slots or numerical capacity can last longer. The
[A100 prefill grouping experiment](baselines/2026-11-20-timesfm-prefill-grouping/README.md)
compares the one-step policy with PR #80 on long bursts and short requests.

### GPU checks

Run separately on an idle Linux GPU: the first command owns a native CUDA model.

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_STEPPING=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_stepping.py -q -s
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_CONTINUOUS=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_continuous.py -q -s
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_PREFILL_HOLD=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_prefill_hold.py -q -s
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_PREFILL_OWNERSHIP=1 VTSFM_PREFILL_GROUP_STEPS=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_prefill_ownership.py -q -s
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

On 2026-11-11, standalone eager commit `b9fffd25` passed 21 pinned-checkpoint
stepper tests, three real HTTP stagger/lifecycle/capacity tests, and six
independent native CUDA versus block-mode HTTP cases. The latter include pinned
AssetOpsBench histories; maximum absolute error was `4.7684e-7` under the
unchanged `rtol=atol=1e-4` gate. These runs used the A100-SXM4-40GB, driver
595.91.07, Torch 2.14.0+cu130, vLLM 0.33.0 V1 and FP32 with TF32 disabled.
TimesFM was pinned to revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`.

An additional test-only commit, `c2b994f6`, compared 16 ragged synthetic histories
of lengths 8,141–8,192 and horizon 1,024 against a separate official native model
with compilation disabled. The unchanged whole runtime matched bitwise. The
stepper's maximum absolute error was `9.5367e-7` for one 16-series owner and
`2.1458e-6` for four four-series owners, across all nine quantiles. Both layouts
executed seven AR calls with 32 internal rows, retained the same persistent KV
bound and released all state. This adds finite full-corner native parity coverage
under the unchanged `1e-4` gate without changing model code.

Separate finite synthetic capacity probes ran four four-series C8192/H1024
requests together, then four 16-series requests with capacity deferral. Both
exercised 16 logical series and 32 internal AR rows, matched isolated block-mode
responses, and retained the **1,861,222,400-byte** persistent KV bound.
Cancellation, response-formatting failure and completion released request state;
all owned HTTP servers stopped.

| Capacity probe | Torch peak allocated | Torch peak reserved | NVML sampled device peak |
| --- | ---: | ---: | ---: |
| Four four-series requests sharing execution | 3.67 GiB | 4.03 GiB | 5.05 GiB |
| Four 16-series requests, one active at a time | 3.14 GiB | 3.48 GiB | 4.51 GiB |

Torch peaks were reset after `empty_cache` outside each probe. NVML sampled
whole-device residency every 20 ms; it is separate from Torch allocator memory
and can miss brief peaks. These finite synthetic measurements cover the tested
shape and lifecycle behavior on this pinned stack; they are not exact total-memory
guarantees. Use these current probes separately from historical baseline results.

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
The standalone path executes native eager math for both whole forecasts and
shared block decoding. Throughput was measured separately below.

## Offered-load comparison

```bash
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 TRITON_DEFAULT_FP_FUSION=0 \
  PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m benchmarks.continuous \
  --data-dir data --output results/continuous-NEW --requests 100 \
  --warmup-requests 100 --warmup-timeout 600 \
  --modes whole,continuous --scenarios saturated --saturated-interval .001 \
  --diagnostic-requests 0
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
small differences. Repeated runner iterations and unequal-capacity attention
calls add overhead. Evaluate the measured tradeoffs against tuned whole batching;
block scheduling alone establishes no speedup.

### Matched eager results

The 2026-11-11 A100 comparison used main model code `045bd2e7` versus the
standalone decoder `b9fffd25`. Baseline checkout `05e08893` adds only the same
measurement patch used by the candidate. The whole-forecast runtime was unchanged.
Both sources used FP32 eager execution, TF32 disabled, four slots, eight fused
logical series, queue limit 128 and four host threads. No Torch compilation was
requested; Dynamo graph and frame counters stayed zero.

Saturation offered identical two-series C512/H1024 AssetOps requests every 1 ms.
Source order was baseline/candidate/candidate/baseline, reversing serving-mode
order between repeats; every mode used a fresh server. Each point completed
100 untimed concurrent warmups, drained engine and worker state, then completed
100 timed requests. All 800 timed requests and 800 warmups completed with zero
failures or rejections. Timing covered HTTP submission through JSON decode;
startup, warmup, numerical checks and allocator RPCs were outside timing.

| Source and mode order | Whole requests/s | Continuous requests/s | Continuous / own whole |
| --- | ---: | ---: | ---: |
| Baseline 1, whole then continuous | 5.183 | 2.345 | 0.452 |
| Candidate 1, continuous then whole | 5.200 | 6.028 | 1.159 |
| Candidate 2, whole then continuous | 5.094 | 5.838 | 1.146 |
| Baseline 2, continuous then whole | 5.179 | 2.407 | 0.465 |

On this workload, candidate continuous throughput was **2.57× and 2.43×** the
baseline continuous throughput, and **15.9% and 14.6%** above its own whole
control. The unchanged whole path varied from 5.09 to 5.20 requests/s; that
variation is not attributed to the block changes. These are repeated homogeneous
sensor queries, not a general performance or forecast-accuracy benchmark.

The four pre-timing case shapes passed cross-source numerical comparison at
`rtol=atol=1e-4`; independent native checks are described above. Timed responses
were checked for HTTP success, request IDs and series counts, without numerical
comparison inside timing. Source checks, exclusive GPU occupancy, warmup/drain
checks and owned-process cleanup passed for all four jobs.

The [November 7 measured baseline](baselines/2026-11-07-continuous/README.md)
records the first complete 900-request comparison of the original block path.
That historical implementation was slower than whole-forecast batching under
homogeneous saturation; it does not measure the decoder described here.

## Experimental Moirai quantile rounds

Moirai can opt into the same synchronous V1 worker, scheduler and block pooler.
The research switch is required in addition to ordinary continuous batching.

```bash
VTSFM_EXPERIMENTAL_MOIRAI_BLOCKS=1 ./scripts/serve.sh --adapter moirai2 --continuous-batching
```

Moirai uses eager FP32, upstream attention, four slots by default and up to eight
active univariate rows in total. The scheduler reserves whole-horizon work under
the ordinary Moirai admission limits. Each virtual marker represents 64 forecast
samples (four patches of 16). The first native module call processes one trajectory
per series. Each subsequent round processes nine per series, then reduces all 81
quantile candidates and feeds the nine resulting trajectories back. The scheduler
can interleave requests between rounds. For each request the model executes rows
grouped by true history length, exactly as in the native reference. Concatenating
rows from independent requests changed FP32 native quantiles over 16 recursive
rounds on the A100, including for equal-length inputs. This prototype therefore
keeps model calls isolated per owner.
There is no attention KV state. Requests own their expanded targets, masks and
output buffers until completion or cancellation. This mode is experimental.
Ordinary Moirai serving still runs complete forecasts. The [A100 benchmark](baselines/2026-11-25-moirai-forecast-blocks/README.md) reports throughput, p95 latency, memory and native parity for both modes.

In the pinned Linux Moirai environment, exercise native parity without occupying
a GPU, or run the HTTP lifecycle and maximum-shape check on an idle A100:

```bash
CUDA_VISIBLE_DEVICES='' VTSFM_MOIRAI_BLOCK_CHECKPOINT=cpu NVIDIA_TF32_OVERRIDE=0 \
  PYTHONPATH=src:tests:. .venv-moirai2/bin/python -m pytest -q tests/test_moirai_stepping.py
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_MOIRAI_CONTINUOUS=1 NVIDIA_TF32_OVERRIDE=0 \
  PYTHONPATH=src:tests:. .venv-moirai2/bin/python -m pytest -q tests/test_moirai_continuous.py
```

The existing open-loop replay accepts `--adapter moirai2`. For example, use the
same prepared MetroPT fixture and reverse the order of `--modes` on a repeat.
The independent-native lifecycle test and the untraced HTTP replay answer
different questions. Benchmarks exclude startup and record failures.

```bash
VTSFM_EXPERIMENTAL_MOIRAI_BLOCKS=1 PYTHONPATH=src:. .venv-moirai2/bin/python -m benchmarks.continuous \
  --adapter moirai2 --data-dir data/model-integration --output results/moirai-block-a \
  --modes whole,continuous --scenarios saturated,bursty --requests 100 --queue-limit 128
```
