# TimesFM numerical admission and execution limits

Whole TimesFM forecasts occupy one vLLM placeholder token regardless of series
count, context, or horizon. Optional block mode uses one marker per 128-point
forecast block. Neither marker count bounds the model's numerical work. The
TimesFM adapter therefore applies a separate policy on vLLM 0.33.0 Model Runner
V1, eager FP32, one GPU. Toto and Moirai do not yet declare validated admission
policies; their measurement and enablement remain in issue #12.

## Default envelope

| Limit | Default ceiling |
| --- | ---: |
| Series per request | 16 |
| Context points per series | 8,192 |
| Forecast horizon | 1,024 |
| Prepared independent series per execution | 16 |
| Attention-work units per model call | 2,580,992 |
| Scheduled requests (`max_num_seqs`) | 4 |
| Shared resident tensor accounting | 32 MiB |

The default launcher still uses one scheduled request and no cross-request
fusion. Enable up to four slots and optional fusion explicitly:

```bash
./scripts/serve.sh --adapter timesfm25 --max-num-seqs 4 --forecast-batch-rows 16
```

The new 16-series admission ceiling is an intentional change from previously
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
and increases above the validated ceilings fail before launch. Increasing the
envelope requires new measurements and an intentional adapter metadata change.

Direct `LLM`/`vllm` users can set the same environment variable before construction
or use adapter defaults. Explicitly set `max_num_seqs` to 1–4. The model checks
slots and resident accounting before checkpoint construction, so bypassing the
launcher does not bypass startup validation. A byte cap too small for the
configured worst corner and slot count is rejected at startup; lower the shape
ceilings or slot count together with it. A tighter work/row limit can also reject
a request that satisfies the separate context/horizon limits.

## Work and memory accounting

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
flip-pass rows. The block decoder combines both AR passes into one
neural batch of `2*R` rows, at most 32 for the unchanged 16-series ceiling.
Prefill retains separate normal and flipped calls. Work already charges
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
four-slot default corner this is **22,020,352 bytes**, below 32 MiB. This bounds
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

The suite checks pre-submission HTTP rejection, exact limit boundaries, and a
shared worker forward whose three accepted requests split into four-row and
two-row model calls under a 200-unit policy. It checks rolling-arrival parity,
then measures the default envelope with Torch peak allocated/reserved counters
and 20 ms NVML sampling. The worker trace/fault-free memory extension is imported
only by this manual test. Evidence JSON and logs are retained in pytest's temp
directory; use a fresh `--basetemp` to choose an artifact directory.

The historical whole-forecast probe on November 7, 2026 used the A100-SXM4-40GB,
driver 595.91.07, Torch
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
The current block probes measured up to 3.67 GiB Torch allocated, 4.03 GiB
reserved and 5.05 GiB sampled device residency, while persistent KV stayed at
1.73 GiB. These finite synthetic probes are not exact memory guarantees across
all inputs, hardware or package versions. NVML sampling can miss brief peaks.
Record a new baseline before raising caps or comparing performance; capacity
checks make no throughput claim.
