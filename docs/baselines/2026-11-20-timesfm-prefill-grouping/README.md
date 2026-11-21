# Bounded prefill grouping for TimesFM

On the archived H1024 burst, one-step grouping raised continuous HTTP
throughput from **8.22 to 10.06 requests/s**, a **22.3% gain** over PR #80.
Whole-forecast reached **10.40 requests/s**. Grouping therefore closed
**84.3% of the gap remaining after PR #80**, leaving **0.34 requests/s**.
The policy is opt-in. [Issue #78](https://github.com/vinli0921/vTSFM/issues/78)

## Workloads and controls

The control uses continuous batching from PR #80 (`9705ff64c`). The
candidate uses the same decoder with `VTSFM_PREFILL_GROUP_STEPS=1`. A
whole-forecast control on the same revision measures the remaining H1024
throughput gap.

| Workload | Offered arrivals per repeat | Question |
| --- | --- | --- |
| Archived W4 | 100 two-series MetroPT forecasts, C512/H1024, spaced 1 ms apart | How much of the H1024 throughput gap does grouping recover? |
| Sparse H96 | 100 single-series Chiller 6 forecasts, C2048/H96, spaced 250 ms apart | Does a lone short request pay additional tail latency? |
| Mixed H96/H1024 | 100 short Chiller 6 forecasts and 28 long MetroPT forecasts, with short requests arriving behind long ones | Does H96 tail latency worsen under contention? |

The arrivals are benchmark load patterns. The histories and native references
come from the pinned AssetOpsBench slices used for the cross-engine benchmark.
The sparse and mixed workloads reuse those histories under new arrival plans.

The mixed plan starts with eight H1024 requests 1 ms apart. At 40, 190, 340
and 490 ms, another five long requests arrive 1 ms apart, followed by 25
short requests spaced 4 ms apart. Both modes replay the same payload bytes,
item IDs and offered offsets. All quantiles are checked against the archived
independent native CUDA forecast after timing.

## Matched HTTP results

| H1024 burst mode | Requests/s by repeat | Pooled requests/s | p95 | p99 | Peak GPU residency |
| --- | --- | ---: | ---: | ---: | ---: |
| Whole-forecast | 10.57, 10.44, 10.21 | 10.403 | 9.109 s | 9.473 s | 2,237 MiB |
| Continuous, PR #80 | 8.22, 8.27, 8.17 | 8.223 | 11.636 s | 12.070 s | 2,349 MiB |
| Continuous, one-step grouping | 10.07, 10.03, 10.09 | 10.061 | 9.471 s | 9.845 s | 2,359 MiB |

| Workload and horizon | PR #80 p95 | Grouped p95 | PR #80 p99 | Grouped p99 |
| --- | ---: | ---: | ---: | ---: |
| Sparse H96 | 99.83 ms | 98.68 ms | 105.12 ms | 108.97 ms |
| Mixed, H96 | 6.955 s | 6.851 s | 7.130 s | 6.951 s |
| Mixed, H1024 | 6.186 s | 5.978 s | 7.132 s | 6.926 s |

For isolated H96 forecasts, p95 improved by 1.15 ms. P99 increased in all
three paired repeats, by 0.09, 9.23 and 1.75 ms respectively. Pooled p99
rose **3.85 ms (3.66%)**. In mixed traffic, H96 end-to-end p95 and p99
fell in all three paired repeats, by **103 ms and 179 ms** in the pooled
comparison.

We repeated the isolated H96 test on the candidate checkout with only the
grouping flag changed. Four fresh servers ran in off/on/on/off order with
100 requests each. All 400 completed and passed parity. Pooled p95 went
from **101.40 to 99.13 ms**, and p99 went from **108.52 to 102.98 ms**.
The isolated p99 regression in the primary runs did not reproduce. Keep
one-step grouping available for long, bursty traffic. These short-request
samples do not establish an isolated p99 guarantee or justify changing the
default for all TimesFM workloads.

All 2,268 timed HTTP requests completed and passed independent native CUDA
parity over all nine quantiles at `rtol=atol=1e-4`, covering 20,722,176
forecast values. Maximum absolute error across the primary workloads was
`9.16e-5`. Both continuous modes used four slots, 16 numerical rows, a
128-request queue, TimesFM 2.5 on
vLLM 0.33.0 V1 and eager FP32 with TF32 disabled. All arms ran on the same
A100-SXM4-40GB with driver 595.91.07. The three repeats rotated the order
whole/control/grouped, control/grouped/whole, grouped/whole/control.
Startup, warmups, traces and parity checks were outside the timed windows.

Throughput divides completions by summed replay wall time. Percentiles pool
raw latencies from offered arrival to decoded HTTP response, including queue
time. There are 300 samples per H96 row per mode, 84 per mixed H1024 row
and 300 per H1024 burst mode. Sparse arrivals never overlapped. The W4
throughput gap fraction is `(grouped − control) / (whole − control)`.
Memory is the highest sampled NVML device residency across repeats. On
W4, peak Torch allocation was 1,143 MiB for PR #80 and 1,179 MiB with
grouping. Peak reservation was 1,296 and 1,306 MiB respectively.

For comparison with the original decoder, three additional W4 runs on
`9cde2f134` reached 7.42, 7.36 and 7.30 requests/s (7.36 pooled).
Against that control the combined PR #80 and #78 changes recovered **88.7%**
of the original throughput gap. These additional runs were outside the
three-way rotation. The 84.3% figure above isolates grouping against PR #80.

## What changed inside the scheduler

In a separate traced W4 replay, the oldest waiting request reached the front
while three active forecasts each had one block left. PR #80 admitted the
single new prefill into the available slot. The one-step policy held it until
the next scheduler iteration, when four compatible requests could prefill
together. This was the only grouping hold in the 100-request replay.

| W4 diagnostic | PR #80 | One-step grouping |
| --- | ---: | ---: |
| Prefill calls | 50 | 26 |
| Prefill group sizes | 25 singles, 25 triples | 1 single, 1 triple, 24 groups of four |
| Decode calls | 200 | 176 |
| Total model calls per completion | 2.50 | 2.02 |
| p95 from EngineCore enqueue to first worker prefill dispatch | 10.87 s | 9.14 s |

All 100 requests reached prefill in each trace. Both preserved FIFO prefill
order. The trace uses host monotonic timestamps. Queuing in this table is
specific to the instrumented burst; the HTTP latency comparison uses
separate untraced samples.

In the separate mixed trace, the one-step policy held one H1024 request and
zero H96 requests. All 128 started prefill in FIFO order in both modes.
Prefill calls fell from 50 to 49, and decode calls from 75 to 74. H96
enqueue-to-prefill p95 **rose** from 6.817 to 6.978 s in this single traced
replay (p99 from 7.089 to 7.123 s). The three untraced mixed replays above
had lower end-to-end H96 tails with grouping. Short requests can still
queue behind earlier long requests even when the policy never holds a short
request directly.

## Cancellation and progress

Real HTTP clients disconnected while the FIFO head was held, three times at
each configured bound (one and two scheduler steps). The active decode advanced
on every hold. All six clients cancelled before prefill, the peers matched
isolated whole-forecast outputs over all nine quantiles, and each cycle
drained the engine and worker. A subsequent lone H1024 request and a lone
H96 request both bypassed the hold.

The existing continuous HTTP suite passed all three tests with one-step
grouping enabled, including staggered joins and completions, active
cancellation, overload, a failed forecast alongside a healthy peer, and
deferral under peak numerical capacity. At the capacity limit, four shared
requests and sixteen deferred requests each peaked at 1,861,222,400 bytes
of persistent K/V. The four-request case reached 5,081,903,616 bytes of
Torch allocation and 6,505,955,328 bytes of sampled device residency.

## Reproduction

On the team A100 host, the following replays all primary arms on an idle GPU.
Use a fresh output directory. The driver checks the source revisions, workload
bytes, checkpoint, environment and GPU ownership before qualifying results.

```bash
exp="$HOME/vtsfm-issue78-20261120"
py="$HOME/vtsfm-upgrade-spike/envs/timesfm25/bin/python"
"$py" "$exp/measure/measure78.py" run \
  --base "$exp/base" --base-commit 9705ff64c0ab11783340bfb938304aa7b04ef543 \
  --candidate "$exp/candidate" --candidate-commit 010e85a56d827b992691808cadfe31935390d816 \
  --candidate-policy-env VTSFM_PREFILL_GROUP_STEPS=1 \
  --archive "$exp/archived-workload" --output "$exp/rerun"
```

The original W4 arrival-plan SHA-256 is
`3a7f93c57fc5d2a34b86961afe10a8df3a6dde68f69877f82e4ff5d917e74c8c`.
`measure/pool78.py` computes pooled percentiles and preserves per-repeat and
per-horizon results. `measure/POOLING.md` includes the analysis command.
The evidence archive on the team GPU host is
`~/vtsfm-issue78-20261120/issue78-prefill-a100-010e85a.tar.gz`, SHA-256
`c34a79b88073b3c3d2360ba9ef647d2d97d7e3cfd1d987263132cce9c7bc7ea5`.
It contains the workload and native reference, raw timings, policy traces,
lifecycle evidence, and the replay and analysis scripts. The GPU process
list was empty after the final run.
