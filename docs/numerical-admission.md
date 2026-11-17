# Numerical admission and execution limits

Whole forecasts occupy one vLLM placeholder token regardless of series
count, context, or horizon. Optional TimesFM block mode uses one marker per 128-point
forecast block. Neither marker count bounds the model's numerical work. The
adapters therefore apply separate policies on vLLM 0.33.0 Model Runner V1,
eager FP32, one GPU. Each descriptor supplies a CPU-only cost-function path and
its own limits; shared code does not borrow another model's formula.

**Validation status:** TimesFM, Toto and Moirai have measured A100 envelopes
below. The November 11 qualification covers native/HTTP parity, actual fused
model calls, boundary/rejection checks, repeated maximum requests and allocator/
NVML peaks. Capacity evidence applies to the recorded pins and settings; it is
not a throughput result or an exact memory guarantee.

## Default envelope

| Limit | TimesFM | Toto | Moirai |
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

`max_execution_rows` counts prepared independent input series, not internal
flip-pass rows. The block decoder combines both passes at prefill and at
each AR step into one neural batch of `2*R` rows, at most 32 for the
unchanged 16-series ceiling. Work already charges
`2*R*cache_capacity^2`, and both owned flip caches were already included in
persistent KV accounting. This changes temporary execution shapes, not request,
row, work or persistent-cache ceilings. Standalone eager commit `b9fffd25`
passed fresh native-parity checks and synthetic capacity probes that exercised
the 32-row internal AR corner. A further test-only commit, `c2b994f6`, compared
16 ragged C8141–8192/H1024 histories with a separate official native model,
using both one 16-series owner and four four-series owners. All nine quantiles
passed the unchanged `1e-4` gate. See the
[current block measurements](continuous-batching.md#checks-and-reproduction).
The November 7 whole-forecast measurements below describe the earlier path.

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

The Toto/Moirai checks use each model's own environment, one adapter
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
metadata dry-runs do not establish GPU compatibility or capacity.

Compact Toto/Moirai public outputs own their result storage, so cropped views
cannot pin a larger native context/forecast grid between requests. Upstream
temporaries still contribute to peak model memory. Toto intentionally retains a
bounded model KV cache across calls; distinguish that high-water allocation from
growing per-request state. Moirai passes the constructor frame explicitly to
Lightning 2.6.6 to avoid a frame cycle retaining forecast inputs and native
outputs until cyclic GC. Its GC-disabled input-lifetime regression and unchanged
strict repeated-wave memory check cover that ownership fix. No per-request
garbage collection or looser tolerance is used.

The TimesFM suite checks pre-submission HTTP rejection, exact limit boundaries, and a
shared worker forward whose three accepted requests split into four-row and
two-row model calls under a 200-unit policy. It checks rolling-arrival parity,
then measures the default envelope with Torch peak allocated/reserved counters
and 20 ms NVML sampling. The worker trace/fault-free memory extension is imported
only by this manual test. Evidence JSON and logs are retained in pytest's temp
directory; use a fresh `--basetemp` to choose an artifact directory.

### TimesFM reference

On November 7, 2026, the A100-SXM4-40GB probe used driver 595.91.07, Torch
2.14.0+cu130, CUDA 13.0, vLLM 0.33.0 V1, eager FP32 with TF32 disabled, and
TimesFM revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`:

| HTTP workload | Peak allocated | Peak reserved | Sampled device peak |
| --- | ---: | ---: | ---: |
| 1 series, C128, H96 | 0.87 GiB | 0.94 GiB | 1.97 GiB |
| 16 series, C8192, H1024 | 2.42 GiB | 2.72 GiB | 3.74 GiB |
| Four maximum requests in one forward, four bounded model calls | 2.43 GiB | 2.72 GiB | 3.74 GiB |

Since #52, whole forecasts decode the normal and flipped passes as one
`[x; -x]` call, so both passes' decode caches and activations are live at once.
The same suite on an RTX A6000 (November 16, 2026; driver 575.57.08 with the
CUDA 13.0 compat libraries, otherwise the stack above) reproduced the rows above
on the previous two-call code (2.43 / 2.72 / 3.64 GiB for the 16-series row).
With the batched call, two runs each measured:

| HTTP workload | Peak allocated | Peak reserved | Sampled device peak |
| --- | ---: | ---: | ---: |
| 1 series, C128, H96 | 0.87 GiB | 0.94 GiB | 1.87 GiB |
| 16 series, C8192, H1024 | 3.66 GiB | 4.16 GiB | 5.08 GiB |
| Four maximum requests in one forward, four bounded model calls | 3.67 GiB | 4.16 GiB | 5.08 GiB |

Work units and row limits are unchanged: they already charge both flips and
count user series.

The historical whole-forecast peaks above do not measure the current block
implementation's temporary attention packs or combined flip-pass batch.
The earlier block probes used separate sign prefills. They measured up to
3.67 GiB Torch allocated, 4.03 GiB reserved and 5.05 GiB sampled device
residency, while persistent KV stayed at 1.73 GiB. Recheck the block-capacity
peak with the batched prefill before using those figures for deployment.
These finite synthetic probes are not exact memory guarantees across
all inputs, hardware or package versions. NVML sampling can miss brief peaks.
Record a new baseline before raising caps or comparing performance; capacity
checks make no throughput claim.


### Toto and Moirai qualification — November 11, 2026

These runs used the same A100-SXM4-40GB, driver 595.91.07, Torch
2.14.0+cu130 / CUDA 13.0, vLLM 0.33.0 V1, Transformers 5.20.2 and eager
FP32 with TF32 disabled. No attention backend, precision or dependency upgrade
was included. Toto used Python 3.12.12 and toto-2 2.0.0; Moirai used Python
3.11.17, uni2ts 2.0.0 and Lightning 2.6.6.

| Adapter | Tested source | Checkpoint revision |
| --- | --- | --- |
| Toto | `e69f2eafce299e8dd89b87498b994aa209734d2e` | `a7bab288f5e95f8606f8306f86659357e1c001ef` |
| Moirai | `b4149d9b1cd81cdf299f1517958bf6b70e35a9df` | `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` |

The later qualification documentation/policy-comment update changes no executable
code. Each run records clean source state, source SHA256s, model pins, launches,
worker configuration, execution traces, parity and cleanup evidence. Admission
artifacts retain native arrays and comparison statistics, rather than complete
HTTP forecast arrays for offline rescoring. Raw files
are retained under the manual run's `gpu-20261111/` directory, referenced in
[PR #55](https://github.com/vinli0921/vTSFM/pull/55).

`C` is history length and `H` is horizon. Peaks below are absolute, not
baseline-subtracted. Torch peaks are reset after `empty_cache` outside each
probe; whole-device NVML is sampled every 20 ms. The four-maximum-request row
reports the maximum across three repetitions. No explicit Python garbage
collection occurs between these measurements.

| Adapter / HTTP workload | Model calls in measured forward | Peak allocated | Peak reserved | Sampled device peak |
| --- | --- | ---: | ---: | ---: |
| Toto: 1×1, C128/H96 | one 1-row call | 1.300 GiB | 1.365 GiB | 2.391 GiB |
| Toto: 1×16, C8192/H1024 | one 16-row call | 2.518 GiB | 2.805 GiB | 3.833 GiB |
| Toto: 4×4, C8192/H1024 | one fused 16-row call | 2.519 GiB | 2.852 GiB | 3.879 GiB |
| Toto: 4×16, C8192/H1024 | four sequential 16-row calls | 2.521 GiB | 3.020 GiB | 4.047 GiB |
| Moirai: 1×1, C128/H96 | one 1-row call | 0.053 GiB | 0.061 GiB | 1.088 GiB |
| Moirai: 1×8, C7168/H1024 | one 8-row call | 1.822 GiB | 2.396 GiB | 3.424 GiB |
| Moirai: 4×2, C7168/H1024 | one fused 8-row call | 1.823 GiB | 2.396 GiB | 3.424 GiB |
| Moirai: 4×8, C7168/H1024 | four sequential 8-row calls | 1.824 GiB | 2.398 GiB | 3.426 GiB |

The fused rows are logical independent series: Moirai's eight logical rows expand
to 72 quantile trajectories on later recursive module calls. Four resident maximum
requests mean bounded sequential model calls, never one 64-row Toto or 32-row
Moirai call.

Both adapters passed four admission tests, including five pre-submission rejection
categories, exact boundaries, actual fusion/work splitting, mixed and rolling
arrivals, and maximum-shape probes. All request/encoder/pooling state cleared.
Post-wave allocated bytes were identical across all three repetitions:
2,336,692,224 for Toto (including retained model KV) and 54,460,928 for Moirai.
The strict 64 KiB stability allowance and native parity tolerances
(`rtol=atol=1e-4`) were unchanged. Maximum saved fused-native absolute errors were
`2.38e-6` for Toto and `1.07e-6` for Moirai; relative errors near zero are interpreted
with the absolute tolerance.

The checkpoint suites passed 31 Toto and 53 Moirai tests, including compact output
storage and Moirai's GC-disabled input-lifetime regression. Six native/HTTP
integration cases per adapter passed, including AssetOpsBench; Moirai's six were
repeated after its frame fix. TimesFM's two tensor-parity, three admission/batching
and three continuous-batching checks also passed. These are correctness/capacity
checks, not observed-label forecast-quality or throughput measurements.

The initial Moirai memory run failed the unchanged stability check. An isolated
GPU diagnostic showed cyclic GC releasing temporary tensors; a tiny actual
upstream CPU oracle then verified the frame fix releases inputs immediately,
preserves `hparams`/`hparams_initial`, and produces bitwise-identical nine-channel
forecasts. The failed run and diagnostic are retained alongside the passing run.

The highest sampled device peaks leave roughly 36 GiB of headroom on this
otherwise idle 40 GiB A100. Keep the declared caps until new measurements
qualify a larger envelope.
