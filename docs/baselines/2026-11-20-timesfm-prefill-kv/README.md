# TimesFM shared prefill K/V storage on the A100

Bounded packing improved H1024 burst throughput **4.52%** and mixed H96/H1024
throughput **4.94%** across four paired repeats. All six retirement/reuse
cycles preserved native forecast parity. The change used 30 MiB more peak
device residency in the timed runs and matched the original peak at the
maximum admitted shape.

This experiment compares the per-layer clone path at `7a8b54225` with bounded
per-request packing at `deda20e87` on the same TimesFM continuous server. Both runs enable
one-step prefill grouping and the PR #80 decode K/V optimization.
[Issue #76](https://github.com/vinli0921/vTSFM/issues/76) ·
[Earlier W4 prototype comparison](https://github.com/vinli0921/vTSFM/issues/51#choosing-between-76-and-79-for-the-remaining-h1024-gap)

For each request in a shared prefill, each flip stream takes its share of the
temporary batched K/V across all 20 layers. The original code cloned each
layer's K, V, write positions and mask counts separately. The candidate
packs the retained K/V slices into one allocation, with one additional buffer
for positions and one for mask counts. Each layer uses disjoint contiguous
views. Shorter horizons retain only their own cache capacity. Separating the
two counters preserves the integer types returned by native prefill. Packing
applies through **18,022,400 bytes per owner per flip**, the measured W4 K/V
allocation. Each larger owner uses the previous per-layer clone path
independently, including when a smaller peer in the same prefill can pack.

## Workloads

The untraced HTTP comparison runs on a single A100-SXM4-40GB, driver 595.91.07,
in the pinned TimesFM 2.5 FP32 environment with vLLM 0.33.0 V1, Torch
2.14.0+cu130 and TF32 disabled, using
four slots, a 16-row execution cap and a 128-request queue. Both arms replay
the same AssetOpsBench sensor histories and native CUDA references. Each
repeat starts a fresh server and excludes startup, warmup and parity checks
from timing.

| Replay | Requests per arm per repeat |
| --- | --- |
| W4 | 100 two-series MetroPT requests at C512/H1024, offered 1 ms apart |
| Sparse H96 | 100 single-series Chiller 6 requests at C2048/H96, offered 250 ms apart |
| Mixed | 28 long MetroPT and 100 short Chiller 6 requests arriving in staggered waves |

The mixed HTTP replay uses two context buckets. A separate lifecycle test
exercises requests that share a padded context and have different row counts
and horizons. Throughput divides completions by total replay wall time.
Percentiles pool the time from offered arrival to decoded response, including
queueing. NVML samples whole-device residency. Torch peak allocated and
reserved bytes come from the worker allocator.

## Untraced HTTP performance

| Workload | Original requests/s | Packed requests/s | Change | Original peak GPU | Packed peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1024 burst | 9.859 | 10.304 | +4.52% | 2,359 MiB | 2,389 MiB |
| Mixed H96/H1024 | 16.814 | 17.644 | +4.94% | 2,359 MiB | 2,389 MiB |
| Sparse H96 | 4.025 | 4.025 | <0.01% | 2,359 MiB | 2,389 MiB |

| Workload and horizon | Original p95 | Packed p95 | Original p99 | Packed p99 |
| --- | ---: | ---: | ---: | ---: |
| H1024 burst | 9.639 s | 9.269 s | 10.031 s | 9.637 s |
| Mixed, H96 | 6.932 s | 6.592 s | 7.108 s | 6.688 s |
| Mixed, H1024 | 6.184 s | 5.838 s | 6.989 s | 6.683 s |
| Sparse H96 | 100.12 ms | 100.81 ms | 108.03 ms | 107.35 ms |

The four paired repeats used original/packed, packed/original, packed/original,
original/packed server order. Each paired run used the same point order.
All 2,624 timed requests completed and 20,256,768 forecast values passed
independent native CUDA parity across nine quantiles (maximum absolute error
`9.16e-05`, `rtol=atol=1e-4`).
H1024 burst rates by repeat were 9.70, 9.96, 9.89, 9.90 requests/s for the original path and 10.16, 10.45, 10.34, 10.27 for the packed path.
Mixed H96/H1024 rates by repeat were 16.45, 16.93, 16.88, 17.01 requests/s for the original path and 17.44, 17.81, 17.74, 17.59 for the packed path.
Sparse H96 arrivals never overlapped. Singleton prefill adopts its cache on
both paths. Four repeats of these fixed workloads support descriptive comparisons.

Peak Torch allocation on W4 and mixed traffic was 1,179.4 MiB for the
original path and 1,186.6 MiB with packing. Peak reserved memory went from
1,306 to 1,336 MiB. Sparse H96 peak allocation was 933.4 MiB in both modes.
Each server ran all three workloads, so point-level residency includes any
allocator reservation carried over from its earlier point.

## Where the copy work went

Two separate stock-NVTX Nsight Systems replays profiled the same W4 requests,
one per revision. Both completed 100 requests with 26 prefill calls and 176
decode calls. The table reports inclusive host time and CUDA operations
attributed to each scope, amortized per completion.

| W4 measure | Original | Packed |
| --- | ---: | ---: |
| Prefill `_store` host time | 4.55 ms | 1.28 ms |
| Prefill `_store` CUDA copy API calls | 164.40 | 6.00 |
| Prefill `_store` CUDA kernel launches | 6.00 | 11.94 |
| Prefill `_store` CUPTI D2D memcpy bytes | 35,685,350 | 48 |
| Full prefill host time | 29.83 ms | 25.58 ms |
| Full prefill CUDA copy API calls | 190.42 | 32.02 |
| Full prefill CUDA kernel launches | 805.90 | 811.84 |
| Full prefill CUPTI D2D memcpy bytes | 35,685,722 | 419 |

Packing gathers the same K/V values into owned storage using CUDA kernels.
CUPTI counts explicit memcpy bytes in the table and excludes bytes moved
inside kernels. The improvement comes from fewer copy submissions. These
traced host intervals can overlap device activity; the HTTP comparison
uses separate untraced runs.

In a separate isolated probe, four two-row W4 owners required **320 K/V copy
activities** across both flip streams on the original path and **8** on the
packed path. Counter copies fell from 320 to 16. Median untraced storage
time per forecast fell from **2.49 to 0.74 ms**. The same launch counts held
for four owners with row counts 3/1/2/2, horizons 129/1024/96/513 and raw
contexts 481/510/496/512, all padded to C512. Their cache widths were
20/44/16/32 patches, and time fell from **2.53 to 0.88 ms**. Each measure
used eight warmups and 60 synchronized samples. An independent profiling
pass attributed CUDA activities to K/V and counter tensor operations.
Exact checks compared each retained tensor to its original slice and type.
Both layouts retained 144,179,200 K/V bytes per W4 cohort and 81,920,000
bytes per mixed cohort.

## Admission-limit memory

The initial unrestricted version (`883c20c8`) increased peak allocator
reservation and sampled GPU residency by **430 MiB** when four maximum-size
requests ran prefill together. The bounded version (`deda20e87`) restored the
original peak in every column of the two capacity probes.

| At C8192/H1024 and 16 active series | Peak retained K/V | Torch peak allocated | Torch peak reserved | NVML sampled peak |
| --- | ---: | ---: | ---: | ---: |
| Four four-series requests sharing execution, original and bounded | 1,861,222,400 B | 5,081,903,616 B | 5,402,263,552 B | 6,505,955,328 B |
| Four four-series requests, unrestricted packing | 1,861,222,400 B | 5,082,456,576 B | 5,853,151,232 B | 6,956,843,008 B |
| Four 16-series requests admitted one at a time, all versions | 1,861,222,400 B | 3,934,203,904 B | 4,458,545,152 B | 5,562,236,928 B |

The probe cleared the allocator cache and reset Torch peaks before each
case. NVML sampled device residency every 20 ms. The limit on packing
controls allocation size and leaves the existing numerical admission
policy in force.

## Ownership under cancellation and slot reuse

A separate A100 test filled both HTTP slots with a shared prefill, queued a
third request, then disconnected or completed one of the first two owners.
Three disconnects used two B2/C512/H1024 owners. Three natural completions
used a ragged two-row H1023 survivor and a one-row C497/H129 peer. A three-row
ragged H255 replacement entered the freed slot in each case.

A test-only worker probe kept weak handles to actual PyTorch storage. In all
six cycles every original storage for the retired owner had expired before
replacement prefill, while the surviving owner still held live storage.
The shared owners and opposite flip streams had disjoint backing allocations.
Logical bytes equaled backing bytes for each owner, including the shorter
H129 cache. All 15 completed HTTP responses matched independent native CUDA
forecasts across all nine quantiles. Maximum absolute error was `1.19e-6`.
The engine drained and the survivor's original storages expired after completion.
All 22 native stepping tests passed on the A100. At C8192/H1024, both the
single 16-row owner and the four four-row owners matched native forecasts
across nine quantiles with a maximum absolute error of `2.74e-6`.
The three existing continuous-serving tests also passed, covering staggered
arrivals, cancellation, failure isolation and the numerical admission limit.

## Reproduction

The team A100 host holds the harness, original W4 payloads and independent
native references. Use an idle GPU. The benchmark controller verifies clean
source revisions, the worker import path, payload hashes, device identity,
precision and cleanup for every point. A fresh experiment directory preserves
the recorded runs.

```bash
run="$HOME/vtsfm-issue76-20261120"
prior="$HOME/vtsfm-issue78-20261120"
py="$HOME/vtsfm-upgrade-spike/envs/timesfm25/bin/python"
mkdir "$run/repeat"
NVIDIA_TF32_OVERRIDE=0 "$py" "$run/perf76.py" \
  --prior "$prior" --experiment "$run/repeat" --base "$run/base" \
  --candidate "$run/final" --candidate-commit deda20e875fb3ab886e3f0ece50bc11789638853
```

The archived load is identical to the H1024 burst profiled for PRs #80 and
#81. Its arrival-plan SHA-256 is
`3a7f93c57fc5d2a34b86961afe10a8df3a6dde68f69877f82e4ff5d917e74c8c`.

The raw timings, reference outputs, CUDA profiles, storage and capacity
probes, source bundles and replay scripts are in
`hpml:~/vtsfm-issue76-20261120/issue76-timesfm-prefill-kv-deda20e-a100.tar.gz`.
Archive SHA-256 is `06f79b830118d9e729049d2ee8235df7833a0af5a349e6480776b516faa196b0`.
The earlier whole-forecast controls are recorded in issue #51. This
two-arm run measures the incremental gain over grouped continuous serving.
