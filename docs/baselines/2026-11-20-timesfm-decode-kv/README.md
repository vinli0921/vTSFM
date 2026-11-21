# TimesFM H1024 decode cache benchmark

Fusing decode K/V appends and packing, together with batching counter updates
per owner, raised continuous HTTP throughput from **7.64 to 8.24 requests/s**
on the archived MetroPT burst. That is a **7.8% throughput gain** and closes
**21.4% of the gap** to whole-forecast on the same GPU. [Issue #77](https://github.com/vinli0921/vTSFM/issues/77)

## Matched HTTP results

| Mode | Requests/s per repeat | Pooled requests/s | p95 offered-to-completion | p99 offered-to-completion | Peak GPU residency |
| --- | --- | ---: | ---: | ---: | ---: |
| Whole-forecast | 10.30, 10.56, 10.45 | 10.44 | 9.10 s | 9.47 s | 2,237 MiB |
| Continuous, baseline | 7.62, 7.70, 7.59 | 7.64 | 12.56 s | 13.02 s | 2,317 MiB |
| Continuous, optimized | 8.29, 8.21, 8.21 | 8.24 | 11.65 s | 12.07 s | 2,319 MiB |

Each repeat used the same 100 HTTP payloads and request IDs, offered 1 ms apart.
Each request carried two 512-observation histories and returned 1,024 steps
for all nine quantiles. Every fresh server warmed up on three requests and
drained before timing. The order rotated across repeats: whole/baseline/candidate,
baseline/candidate/whole, candidate/whole/baseline. All 900 requests completed,
and all passed independent native CUDA parity at `rtol=atol=1e-4`. Maximum
absolute error across responses was `1.67e-6` for every mode.

Throughput is 300 completions divided by the sum of the three untraced replay
durations per mode. Latencies are pooled over each mode's 300 responses and
include queueing from the scheduled arrival through JSON decoding. Peak
residency is the largest NVML sample across repeats. The worker's peak Torch
allocation was 1,198,722,560 bytes before and 1,198,693,888 bytes after the
change. Both reserved 1,327,497,216 bytes at peak. Gap closure is
`(candidate − baseline) / (whole − baseline)` in requests/s. The remaining
throughput difference is **2.20 requests/s** on this uniform overload burst.

Both controls used clean `9cde2f134` and the candidate used clean `7b9415619`.
All nine runs used the same A100-SXM4-40GB, driver 595.91.07, vLLM 0.33.0 V1,
Torch 2.14.0+cu130, eager FP32 with TF32 disabled, four slots, 16 admitted
execution rows and a 128-request queue. The physical GPU UUID differed from
the November 18 profile. The current comparison uses fresh controls on the
replacement A100. All services were checked for foreign GPU processes and
released the GPU after their run. [Original W4 profile](https://github.com/vinli0921/vTSFM/issues/51#why-whole-forecast-won-the-timesfm-h1024-burst)

## Where the improvement appeared

On the isolated mixed-position cache operation, the reference issued **16
append kernels and two packing kernels per layer**. The candidate issued
**one** copy kernel. The test used eight owner/flip groups, 16 internal rows,
capacity 44, four-patch writes and 20 layers. Exact comparisons checked both
the gathered tensors and the independently owned caches after poisoning the
write spans.

| Isolated operation | Median untraced wall per layer |
| --- | ---: |
| Reference append only | 284 µs |
| Reference pack only | 35 µs |
| Reference append and pack together | 321 µs |
| Candidate fused append and pack | 54 µs |

Each median covers 300 samples, with 20 iterations per sample. The candidate
also spent **1.15 ms per forward** preparing metadata and scratch for 20 layers.
It still gathers full-width K/V for attention. In this geometry it writes
655,360 new K/V bytes into owned caches and uses 7,208,960 bytes of gather
scratch. Counting tensor reads and writes gives 15,728,640 bytes per layer
for the reference and 15,073,280 for the candidate, excluding pointer
metadata. These are logical bytes. The profiler counted kernel launches and
did not measure DRAM transactions. HTTP results include preparation, counter
updates and the rest of the model.

In a separate fixed-cohort test with all eight input rows already on the GPU
and 24 rotated samples per configuration, median block forecast time with
four independent owners fell from **407.0 to 369.2 ms**. With one owner of all
eight rows it rose from **326.5 to 332.1 ms**. The whole-forecast control in
those two sessions measured **354.6 and 359.1 ms** respectively.

## Lifecycle and memory

The checkpoint test used independent native forecasts for owners writing at
positions 16, 20, 24 and 28 with the same 44-token cache capacity. It checked
all nine quantiles as owners were reordered, one cancelled, one finished early,
and a new owner reused capacity. Maximum absolute error was `1.91e-6`.
The HTTP lifecycle tests passed staggered completion, three disconnect cycles
and a failed response while a peer continued successfully.

The capacity suite measured 1,861,222,400 bytes of persistent K/V at the
admitted limit in both the shared and deferred cases. The four-request shared
case peaked at 5,081,903,616 bytes allocated by Torch and 6,505,955,328 bytes
in NVML. The GPU process list was empty after qualification.

## Reproduction

The team GPU host stores the replay scripts, raw responses' parity records,
request timings, profiler summaries, and qualification logs under
`~/vtsfm-issue77-20261120/`. To replay there, choose a fresh output directory
and run the following with the GPU idle.

```bash
exp="$HOME/vtsfm-issue77-20261120"
base="$HOME/vtsfm-issue51-timesfm-20261118"
py="$HOME/vtsfm-upgrade-spike/envs/timesfm25/bin/python"
"$py" "$exp/measure_w4.py" run \
  --baseline "$base/main" --candidate "$exp/candidate" \
  --candidate-commit 7b941561988710880f65587acf23674c1099d849 \
  --archive "$base/timed/timesfm25" --output "$exp/runs-repeat"
```

The driver checks the pinned checkpoint, package versions, native reference
and payload hashes before serving. The compact arrival-plan SHA-256 is
`3a7f93c57fc5d2a34b86961afe10a8df3a6dde68f69877f82e4ff5d917e74c8c`.

The self-contained evidence archive in the experiment directory is
`issue77-w4-a100-7b94156.tar.gz`, SHA-256
`654607d3bc17391d2feb19a0ff0345fb2680d708b244b1dd0be51eec05827b23`.
It includes the compact original workload, native reference, raw HTTP samples,
cache profiles, lifecycle evidence and the candidate Git bundle.
