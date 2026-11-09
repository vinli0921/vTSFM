# TimesFM continuous-batching measurement — November 7, 2026

This first pass validates the replay harness and measures the experimental
forecast-block path. **It does not establish a speedup over configured
whole-forecast batching.** Continuous batching was substantially slower on
homogeneous saturation. Its small bursty p95 difference is inconclusive.
The independent HTTP tests establish staggered admission, shared AR execution,
independent completion and cancellation; those are separate correctness results.

## Configuration and provenance

- A100-SXM4-40GB, UUID `GPU-45ef44e7-252f-0542-814b-060d1462d01f`, driver 595.91.07.
- vLLM 0.33.0 **V1**, Torch 2.14.0+cu130, CUDA 13.0, eager FP32, TF32 disabled.
  All three arms explicitly disable asynchronous scheduling.
- Model `google/timesfm-2.5-200m-pytorch`, revision
  `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. Installed `timesfm` package 3.0.2
  comes from pinned source `e51928e27119cb17bebc005be2696b75e0a9e688`;
  the model and implementation used here are TimesFM 2.5.
- Arms run sequentially in order **single, whole, continuous**. Single uses one
  slot; whole and continuous use four slots and a fusion cap of eight series.
  The queue cap is 128 waiting/running requests for every arm, rather than the
  continuous server's normal 16 default. Numerical settings and payloads match.
- Four warmup/parity cases precede measurement. Each primary scenario has 100
  offered requests; all 900 primary requests completed, with zero errors or
  rejections. Scheduler/worker trace writes are disabled during primary timing.
- Exact inputs, arrivals, commands, stack, GPU and source hashes are saved in
  `bench-final-100/manifest.json` and `summary.json`. The run records base commit
  `31758b82abff7dc11b828cddbf614697a5e9208e` plus working changes and the hashes
  of the actual feature sources. All 50 production source hashes match the
  final local tree. The measured benchmark script hash, verified against both
  the GPU checkout and local copy, is:
  `6c06a037dada8beb869e855ea4d8823b4411ad549ff00f55b8c6a927e3f6cb4a`.

Inputs use the pinned AssetOpsBench MetroPT fixture and its checked causal
forward-fill provenance. The four cases use one/two univariate series, C128/C512
and H128/H513/H129/H1024, all nine quantiles, at the same source origin 4092.
Their histories include respectively one or 54 causally filled time positions.
These are repeated, correlated windows from one compressor. Future labels are
not scored: the fixture lacks a fully observed 513/1024-step holdout. This is a
serving comparison, not an accuracy benchmark.

## Primary measurements

Latency covers pre-encoded HTTP submission through JSON response decode.
Throughput is successful completions divided by replay/drain wall time;
series/s counts each request's actual rows. Sparse requests arrive every second.
Saturation offers homogeneous B2/C512/H1024 requests every 1 ms, allowing the
whole-forecast path to fuse compatible requests. Bursts contain 16 arrivals
every 250 ms and interleave case indices `(3,0,3,1,2,1,2,0)`; exact assignments
and offsets are identical across arms.

| Load | Arm | Requests/s | Series/s | p50 (s) | p95 (s) | Sampled device peak (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sparse | Single | 1.002 | 1.503 | 0.332 | 0.753 | 1.987 |
| Sparse | Whole | 1.003 | 1.504 | 0.338 | 0.751 | 1.987 |
| Sparse | Continuous | 1.002 | 1.503 | 0.365 | 0.812 | 2.045 |
| Saturated | Single | 1.378 | 2.756 | 36.569 | 68.820 | 1.987 |
| Saturated | Whole | 5.372 | 10.744 | 10.033 | 17.785 | 2.065 |
| Saturated | Continuous | 2.441 | 4.881 | 21.410 | 39.354 | 2.221 |
| Bursty | Single | 2.683 | 4.051 | 18.128 | 33.880 | 1.987 |
| Bursty | Whole | 2.820 | 4.258 | 18.513 | 33.030 | 2.086 |
| Bursty | Continuous | 2.763 | 4.173 | 19.311 | 32.747 | 2.223 |

Relative to whole forecasting, continuous saturation throughput is **54.6%
lower** and p95 is **121.3% higher**. Sparse throughput is arrival-limited;
continuous p95 is 8.1% higher. Bursty throughput is 2.0% lower, while p95 is 0.86%
lower. That last difference should not be called an improvement from one pass.
Per-case burst medians also vary in direction; the aggregate does not establish
a broad short-request benefit.

Torch allocated/reserved peaks are retained separately in the raw point files.
The table reports whole-device NVML residency sampled every 20 ms, which includes
allocator reservation and driver/context memory and can miss short transients.
These workloads do not exercise the maximum persistent-cache envelope; see the
[implementation guide](../../continuous-batching.md) for that separate probe.

## Queue diagnostics and limits

Queue spans come from separate traced 24-arrival prefixes using the same
settings and request order. All 24 completed requests per diagnostic point have
correlated enqueue/schedule events. This **engine-only initial queue delay** is
enqueue to first scheduled block; it excludes HTTP parsing/preprocessing and
waits between later blocks. It is not total server waiting time.

For the saturated diagnostic prefix, mean initial queue delay was 7.506 s
(single), 1.712 s (whole) and 4.049 s (continuous). Do not subtract these prefix
diagnostics from the untraced 100-request primary latencies. The primary
Prometheus queue mean/count is also retained, with 100 recorded observations
per point, under vLLM's own metric definition.

This is **one ordered pass, 100 observations per point**, without independent
repeats or reversed arm order. p95 depends on roughly five tail observations;
there are no confidence intervals. Client dispatch lag is recorded separately
(p95 below 15 ms); no arrival is held until an earlier response completes.
The report supports a functional milestone and a measured regression to profile,
not a conference novelty or general performance claim. Cache repacking and
extra engine iterations are investigation targets, not attributed causes.

Reproduce with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=src:tests:. \
  .venv-timesfm25/bin/python -m benchmarks.continuous \
  --data-dir /path/to/prepared/assetops --output results/continuous-NEW \
  --requests 100 --queue-limit 128 --sparse-interval 1.0 --diagnostic-requests 24
```

Raw run directory on the GPU node:
`~/vtsfm-continuous15-20261107/bench-final-100/`. Keep raw samples, source
provenance and traces in ignored artifacts; this document is the reviewed
summary. Future performance claims need repeated/counterbalanced runs and
profiles that explain the saturated regression.
