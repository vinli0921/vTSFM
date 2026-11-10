# Numerical admission and execution limits

Whole forecasts occupy one vLLM placeholder token regardless of series
count, context, or horizon. Optional TimesFM block mode uses one marker per 128-point
forecast block. Neither marker count bounds the model's numerical work. The
adapters therefore apply separate policies on vLLM 0.33.0 Model Runner V1,
eager FP32, one GPU. Each descriptor supplies a CPU-only cost-function path and
its own limits; shared code does not borrow another model's formula.

**Validation status:** TimesFM has the measured A100 envelope below. Toto and
Moirai currently have candidate limits and manual validation tests; their GPU
memory, real HTTP fusion and parity checks are pending an idle node. These
candidates must not be described as measured capacity until that evidence exists.

## Default envelope

| Limit | TimesFM | Toto candidate | Moirai candidate |
| --- | ---: | ---: | ---: |
| Series per request | 16 | 16 | 8 |
| Context points per series | 8,192 | 8,192 | 7,168 |
| Forecast horizon | 1,024 | 1,024 | 1,024 |
| Rows per model call | 16 | 16 | 8 |
| Model-specific work units per call | 2,580,992 | 1,401,856 | 285,834,752 |
| Scheduled requests (`max_num_seqs`) | 4 | 4 | 4 |
| Shared resident tensor accounting | 32 MiB | 32 MiB | 32 MiB |

Work units have different definitions across families; compare budgets within
one adapter, not between models. The new Toto/Moirai row caps intentionally
replace previously uncapped requests. They retain their full declared context
and horizon ranges. Model-owned scratch and caches are separate from the shared
32 MiB bound.

The default launcher still uses one scheduled request and no cross-request
fusion. Enable up to four slots and optional fusion explicitly:

```bash
./scripts/serve.sh --adapter timesfm25 --max-num-seqs 4 --forecast-batch-rows 16
./scripts/serve.sh --adapter toto2 --max-num-seqs 4 --forecast-batch-rows 16
./scripts/serve.sh --adapter moirai2 --max-num-seqs 4 --forecast-batch-rows 8
```

The declared per-adapter row ceiling is an intentional change from previously
uncapped requests. Larger offline/HTTP requests are rejected before allocating
their context tensor or submitting work to EngineCore. Validation errors are
HTTP 400 with the exceeded limit. Offline calls raise `ValueError`.

## Tighten the policy

Pass a partial JSON file; omitted values retain their defaults:

```json
{
  "max_request_rows": 8,
  "max_context": 2048,
  "max_horizon": 512,
  "max_execution_rows": 8,
  "max_execution_work": 200000,
  "max_scheduled_requests": 4
}
```

```bash
./scripts/serve.sh --adapter timesfm25 --max-num-seqs 4 \
  --forecast-batch-rows 8 --forecast-limits limits.json --dry-run
```

Remove `--dry-run` to serve. The resolved JSON appears as
`VTSFM_FORECAST_LIMITS`; the launcher reads the file once and sends the same
immutable policy to frontend and worker. It clears an inherited policy from a
different server. Unknown fields, duplicate keys, noninteger/negative values,
and increases above the declared ceilings fail before launch. Increasing the
envelope requires new measurements and an intentional adapter metadata change.

Direct `LLM`/`vllm` users can set the same environment variable before construction
or use adapter defaults. Explicitly set `max_num_seqs` to 1–4. The model checks
slots and resident accounting before checkpoint construction, so bypassing the
launcher does not bypass startup validation. A byte cap too small for the
configured worst corner and slot count is rejected at startup; lower the shape
ceilings or slot count together with it. A tighter work/row limit can also reject
a request that satisfies the separate context/horizon limits.

## Work and memory accounting

### TimesFM

For padded context length `C` and horizon `H`, pinned TimesFM uses
`P = ceil(C/32)` context patches and `D = floor((H-1)/128)` later decode steps.
Each later step feeds back four patches. The work proxy is:

```text
work_units = 2 * series_count * (P + 4*D)^2
```

The factor two counts flip-invariance decode passes. This is a conservative
attention-position proxy, not an exact FLOP count or memory estimate. Ragged
histories are charged at their longest padded width. All nine native quantiles
are computed regardless of the requested subset. The prefill quantile head
also computes more values than the cropped public output; its scratch memory
must be measured separately.

### Toto

Toto keeps its pinned 32-point patch and 768-point decode-block policy. Let
`P=ceil(C/32)`, `N=ceil(H/32)` and `B=min(N,24)`. The first temporal attention
call costs `(P+B)^2`. Each later block of `b` patches, after `n` patches have
been predicted, adds `(B+b)*(P+n+b)`. Multiply the sum by logical series count.
This charges both the previous block's median feedback and new prediction
positions; H768 and H769 cross a real additional-call boundary.

Upstream retains model-owned KV capacity of `P+2*N` positions and rounds native
forecast buffers to 32 points. Those buffers, caches and head/scaler temporaries
need GPU measurement; the public result accounting stays `[rows,9,H]` FP32.

### Moirai

For patch size 16 and output block 64, let `P=ceil(C/16)`, `F=ceil(H/16)`,
`K=ceil(F/4)` and `T=P+F`. Every module call sees the full past/future sequence.
The first uses `rows` samples; later passes use nine quantile trajectories each:

```text
work_units = rows * ((1 + 9*(K-1))*T^2 + 81*max(0,F-4)*16)
```

The final term charges recursive quantile-candidate reduction. This is not
median-only feedback. Full-sequence projection/scaling, masks/biases and
recursive trajectory storage are model scratch; fused attention is not assumed.
Ragged histories are charged at their longest padded width. The default maximum
corner expands eight logical rows into 72 trajectory rows on recursive calls.

### Shared execution and storage

Frontend admission checks both request and standalone execution limits. For
compatible dense requests, the worker computes the effective fusion row cap as
the minimum of `--forecast-batch-rows`, the execution-row limit, and
`floor(max_execution_work / work_per_row)`. The existing FIFO planner splits work
into sequential model calls. Ragged or fusion-oversized requests retain isolated
execution, but must already satisfy the standalone limits. Individually valid
requests are not rejected merely because they arrive together.

The shared resident bound counts twice the input bytes, three times actual
output bytes, a rectangular output pack using the independent maximum rows and
horizon, and two maximum metadata/header allowances per request. For the
four-slot TimesFM/Toto default corner this is **22,020,352 bytes**; Moirai's
eight-row corner is **14,942,464 bytes**, both below 32 MiB. This bounds
shared staging, retained results, cropping/packing and envelope copies. It
**excludes model weights, upstream attention caches/scratch, CUDA context and
allocator reservation**. The startup slot check makes this bound conservative
for every supported mixture; no expected overload rejection is deferred until
the live worker, where a V1 exception could terminate the engine.

This is a GPU execution envelope. Whole-forecast mode does not bound queued
CPU payloads unless `--forecast-queue-limit` is set. Optional
[forecast-block scheduling](continuous-batching.md) bounds API intake by default
and reserves these row/work limits across **all active requests together**.
Its persistent two-pass KV caches have a separate measured memory envelope;
the 32 MiB staging bound excludes those caches. Cancellation and restart
semantics are described in [request lifecycle](request-lifecycle.md).

## Measurement and reproduction

Run the opt-in suite on an idle Linux GPU:

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_ADMISSION=1 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_admission.py -q -s
```

The candidate Toto/Moirai checks use each model's own environment, one adapter
per fresh process:

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ADMISSION_ADAPTER=toto2 PYTHONPATH=src:tests:. \
  .venv-toto2/bin/python -m pytest tests/test_model_admission.py -q -s
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ADMISSION_ADAPTER=moirai2 PYTHONPATH=src:tests:. \
  .venv-moirai2/bin/python -m pytest tests/test_model_admission.py -q -s
```

These test pre-allocation/offline guards and pre-submission HTTP rejection,
exact boundaries, work-limited splitting with real fused calls, isolated/mixed
parity, repeated arrivals, and maximum-shape allocator/NVML memory. They refuse
foreign GPU workloads. Use a fresh `--basetemp` to retain evidence. CPU tests and
metadata dry-runs do not establish the candidates' GPU compatibility or capacity.

Compact Toto/Moirai public outputs own their result storage, so cropped views
cannot pin a larger native context/forecast grid between requests. Upstream
temporaries still contribute to peak model memory. Toto intentionally retains a
bounded model KV cache across calls; distinguish that high-water allocation from
growing per-request state.

The suite checks pre-submission HTTP rejection, exact limit boundaries, and a
shared worker forward whose three accepted requests split into four-row and
two-row model calls under a 200-unit policy. It checks rolling-arrival parity,
then measures the default envelope with Torch peak allocated/reserved counters
and 20 ms NVML sampling. The worker trace/fault-free memory extension is imported
only by this manual test. Evidence JSON and logs are retained in pytest's temp
directory; use a fresh `--basetemp` to choose an artifact directory.

On November 7, 2026, the A100-SXM4-40GB probe used driver 595.91.07, Torch
2.14.0+cu130, CUDA 13.0, vLLM 0.33.0 V1, eager FP32 with TF32 disabled, and
TimesFM revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`:

| HTTP workload | Peak allocated | Peak reserved | Sampled device peak |
| --- | ---: | ---: | ---: |
| 1 series, C128, H96 | 0.87 GiB | 0.94 GiB | 1.97 GiB |
| 16 series, C8192, H1024 | 2.42 GiB | 2.72 GiB | 3.74 GiB |
| Four maximum requests in one forward, four bounded model calls | 2.43 GiB | 2.72 GiB | 3.74 GiB |

The observed peak leaves substantial headroom on this idle A100. These are
finite synthetic shape probes, not an exact memory guarantee across all inputs,
hardware, package versions, concurrent external workloads, or future kernels.
NVML sampling can miss brief transients. Record a new baseline before raising
caps or comparing performance. This change makes no throughput speedup claim.
