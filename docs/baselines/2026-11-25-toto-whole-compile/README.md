# Whole-forecast compilation for Toto on the A100

Compiling Toto's complete numerical forecast improved warm HTTP throughput
by **36–43% versus optimized eager** on two measured H96 AssetOpsBench shapes. Both reversed runs improved in all
four workload/client combinations. The first forecast after readiness took
**31.05–31.66 seconds** for the whole-forecast compiler, versus **0.37 seconds**
for eager. A newly seen second shape still took 20.97–21.72 seconds to compile.
Keep whole compilation opt-in. It increased native parity failures on a
high-dynamic-range condenser-flow signal, as detailed below.

## Methods

The three arms used the same [main baseline](https://github.com/vinli0921/vTSFM/commit/2e13348),
with the candidate and instruments at `6ddddeff15dea3acd90c11597a9a93507ddb8cbf`.
The benchmark ran through vLLM 0.33.0 V1 pooling, Torch 2.14.0+cu130,
Transformers 5.20.2 and the pinned Toto 2 313M checkpoint `a7bab28`, on one
A100-SXM4-40GB. Every worker selected FP32 with TF32 disabled. All arms used the
two validated fast unit-scaling shims. The compiler arms used identical
Inductor options with CUDA Graphs disabled. vLLM's own compiler stayed off in
all three arms. The **core** arm is the
shipped configuration. The **whole** arm includes masked input preparation,
native forecast iterations and quantile formatting, with no additional eager
submodule exclusions.

Each HTTP arm started a fresh one-slot synchronous worker. Each block used one
MetroPT window per shape, all nine quantiles, identical source series and
pre-encoded payloads across arms. The fixtures come from the prepared
AssetOpsBench data. The two blocks used whole/core/eager then eager/core/whole.
The second block also reversed shape order. Each shape ran at one and four closed-loop
clients, with five untimed warmups and 100 timed requests per point per block.
Four clients queued on the same single slot. Latency includes loopback HTTP
and decoded JSON responses. Throughput divides completions by summed wall time
and percentiles pool the raw samples. Time to readiness, the first call per
shape, parity checks and diagnostic RPCs are outside the warm samples.

## Warm serving and cold cost

| Rows / context | Clients | Requests/s eager / core / whole | p50 ms eager / core / whole | p95 ms eager / core / whole | Whole throughput vs eager |
| --- | ---: | --- | --- | --- | ---: |
| 1 / 2048 | 1 | 13.28 / 10.38 / 18.66 | 74.58 / 95.88 / 53.31 | 78.63 / 98.39 / 55.05 | +40.53% |
| 1 / 2048 | 4 | 14.60 / 11.14 / 20.90 | 272.93 / 357.86 / 190.44 | 282.11 / 367.05 / 195.08 | +43.17% |
| 4 / 512 | 1 | 12.98 / 10.19 / 17.70 | 76.72 / 97.93 / 56.06 | 79.03 / 100.39 / 58.54 | +36.43% |
| 4 / 512 | 4 | 14.49 / 10.98 / 20.73 | 272.92 / 362.82 / 192.63 | 285.97 / 373.37 / 196.36 | +43.01% |

Highest sampled warm device residency across both shapes was 2,417 MiB eager,
2,457 core and 2,453 whole. Peak worker allocator values were 1,314, 1,314
and 1,312 MiB, respectively. NVML sampled device residency every 20 ms. Worker
allocator peaks span the warmups and timed requests. Ready times were 33–34 s
for every worker except the first whole worker at 56 s. Disk caches for
Inductor and Triton were fresh per arm.

Every accepted arm completed all 400 timed HTTP requests. Separate checks
compared 25,920 forecast values with independent native CUDA including item
IDs and quantile order, with maximum absolute difference **1.14441e-5**.
All six runs had the same native result for each payload. The worker confirmed
that exactly the selected method had been wrapped. Whole produced multiple
Dynamo graphs. The measured worker reported a per-frame recompile limit of 16.
Counters sampled before warmup and after timing cannot disambiguate the phase
of every specialization. One earlier attempt encountered a foreign GPU process
during setup. Its separate record contains no timed samples.

With one client at B1/C2048, whole compilation saved about 21 ms per warm
forecast. A 31-second cold penalty would take roughly 1,500 subsequent forecasts
at that fixed shape to recover, assuming the worker stays resident and the warm
latency difference persists. The table uses two shapes and repeats each window. Forecasts with new shapes
can incur additional specialization costs. Establishing that cost on
heterogeneous traffic requires a separate arrival and shape distribution.
[Reproduction commands](../../compiled-execution.md#experimental-whole-forecast-scope-for-toto-fp32)

## Numerical qualification

All seven integration checks passed. The six real HTTP comparisons covered
8,586 values across synthetic and AssetOpsBench cases including ragged rows,
sparse quantile selections, a recursive forecast and a repeated short case
after shape changes. The maximum absolute error was 1.14441e-5. The direct
full-checkpoint parity test alternated B1/C512/H96 and B16/C2048/H800 with
changed payloads, checked that the inputs survived each call, and observed 84
new Dynamo graphs. All checks used the existing 1e-4 absolute and relative
tolerances.

The five MetroPT accuracy windows each used seven series, a 128-step context
and a 32-step horizon. All 10,080 quantile checks passed across 1,120 observed
targets. The means across windows for native / whole were MAE
0.61040077 / 0.61040073, MASE 1.23850196 / 1.23850192 and scaled quantile
loss 1.13103714 / 1.13103710.

The separate [five-signal AssetOpsBench stress set](../2026-11-14-assetops-signal-accuracy/README.md)
contains 2,400 observed labels and 21,600 quantile values per arm, at B5/C512/H96.
Both optimized eager and whole failed strict parity on condenser flow. Of its
4,320 quantile cells, eager missed 69 and whole missed 110. The sets shared 39
cells, with 30 failures specific to eager and 71 specific to whole. The other
four signals passed in both arms. Every reference and payload matched across
arms. The largest absolute difference across all condenser cells was 0.00390625
for each arm. All failed outputs had native magnitude below 10.42 on series
with forecasts reaching 10,118. Across the five cases, mean native / whole
MASE was 1.91795073 / 1.91795072 and SQL was 1.35463150 / 1.35463154.
The original tolerance remains in force.

## Full admission envelope

All four tests passed on the same candidate. Five invalid requests were
rejected, and four eligible requests with four rows each fused into one call
at C8192/H1024, with maximum absolute native difference 2.5034e-6. Three
additional waves each queued four maximum B16/C8192/H1024 requests. After
every wave, request state and slots were empty and allocated bytes were
identical at 2,336,692,224. The fused call peaked at 2,681.76 MiB allocated.
The first four-maximum wave had the highest reserved and sampled device
peaks at 3,148 MiB and 4,200.56 MiB respectively.

The admission harness has a legacy fixed “eager FP32” label in its JSON.
The recorded launch environment selected whole compile, and actual worker
counters accumulated 30, 82, 126 and 128 graphs before leveling off. The
worker logged eager fallback for some frames after 16 specializations on
heterogeneous geometries. The envelope tests establish routing, numerical
behavior on their synthetic inputs and memory cleanup under that fallback.

## Fixed-shape replay diagnostic

A separate in-process probe traced B1/C2048/H96 before attempting explicit CUDA
Graph capture. The unchanged eager and whole paths produced 2,788 and 1,013
CUDA activities, respectively. Both then failed to capture at the same line.
Toto's fast-scaling shim compared a tensor-valued forward scale to 1.0 in a
Python conditional. Reading that CUDA scalar forced a host synchronization
during capture.

A diagnostic kept tensor-valued forward scales in a device multiplication and
preserved the Python-scalar fast path. For both modes, ordinary and replayed
outputs matched an independent native reference on A/B/A inputs including
previously returned output buffers. Four direction-balanced blocks produced
100 timed samples with each execution method. Each sample copied the input
and mask, called the ordinary or captured forecast, cloned its output, and
synchronized. The profiler ran outside timing.

| Runtime with tensor scales kept on device | Ordinary p50 / p95 ms | CUDA Graph p50 / p95 ms | Capture time |
| --- | --- | --- | ---: |
| Eager | 80.36 / 85.09 | 11.15 / 14.11 | 89.9 ms |
| Compiled whole | 11.57 / 12.49 | 6.79 / 6.85 | 23.3 ms |

The direct compiled probe went from 30 unique Dynamo graphs and 457 captured
calls with the original shim to three graphs and 7,365 captured calls with the
diagnostic. Both captures increased live allocator bytes by 8.13 MiB across
warmup and capture. This measurement excludes reserved memory. These rows
exercise one serialized shape and two finite payloads. The serving benchmark
above runs the original shim. Integrating this experiment with HTTP would
require an owned graph lifetime, a fallback for unsupported requests, and
fresh held-out numerical and four-slot capacity tests.

## Raw records

The reference host stores the raw HTTP evidence at
`~/vtsfm-final-20261125/toto/toto-http-6ddddef.tar.gz`. Its SHA-256 is
`e8b56cab6889dbf7828cbd6672ff879ee39c926a1e87a721498dc5abfdf9556b`.

The combined archive includes the HTTP runs, qualification, both failed
captures, raw per-sample replay timings, probe source and command scripts.
It is available at `~/vtsfm-final-20261125/toto/toto-qualified-6ddddef.tar.gz`
on the reference host. SHA-256
`0f773fa5bcd2696e84cbaafc84775d1475779264d101df49fb60668e79de735a`.
