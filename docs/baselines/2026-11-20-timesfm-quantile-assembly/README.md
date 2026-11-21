# TimesFM whole-forecast and continuous output assembly on the A100

## Fresh comparison before #79

Both serving modes ran at the same [PR #82 revision](https://github.com/vinli0921/vTSFM/commit/9b35fbc8e191cf5dedb538c97063581be81b6943). Four paired repeats used whole/continuous, continuous/whole, continuous/whole and whole/continuous order.

| Mode | Requests/s | p95 | p99 | Peak device residency |
| --- | ---: | ---: | ---: | ---: |
| Whole-forecast | 10.316 | 9.166 s | 9.589 s | 2,237 MiB |
| Continuous with packed prefill | 10.242 | 9.337 s | 9.698 s | 2,365 MiB |

Whole-forecast led by **0.074 requests/s (0.72%)** overall. Its per-pair leads were +0.301, +0.214, −0.413 and +0.204 requests/s. That spread motivated a second experiment with the candidate and both controls in the same rotation. All 800 forecasts passed native CUDA parity across nine quantiles.

## What #79 changes

TimesFM predicts a 128-step block at a time. Each sign-flipped pass already keeps its median for the next autoregressive step. The candidate writes the difference between those two medians into the request's own output buffer. On completion it combines all saved medians with the prefill spreads and applies the crossing correction, denormalization and positivity constraint once across the requested horizon. A short final block uses its actual length. Canceling between blocks releases the buffer and skips final assembly.

The tested runtime is `ed5e9f2802476b2abc5bee24737e7facd29fd43c`. The change needs no configuration flag or checkpoint change.

## Three-way W4 comparison

Each mode ran once per rotation in all six possible orders. All **1,800** timed requests completed with zero failures or rejections. Every response passed the independent native CUDA check at all nine quantiles (`rtol=atol=1e-4`, maximum absolute error `1.67e-6`).

| Mode | Requests/s | p95 | p99 | Peak device residency | Peak Torch allocated | Peak Torch reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole-forecast (#82) | 10.395 | 9.116 s | 9.490 s | 2,237 MiB | 1,062.5 MiB | 1,186 MiB |
| Continuous (#82) | 10.377 | 9.184 s | 9.547 s | 2,365 MiB | 1,186.6 MiB | 1,312 MiB |
| Continuous with #79 | **10.558** | **9.019 s** | **9.378 s** | 2,365 MiB | 1,186.5 MiB | 1,312 MiB |

Moving assembly to completion increased throughput **1.74%** over #82 continuous and **1.57%** over whole-forecast. It lowered p95 by 165 ms and 97 ms respectively. The candidate was faster than each control in five of six rotations. The paired candidate-minus-#82 differences were **+0.122, +0.146, +0.292, −0.169, +0.409, +0.300 requests/s**. These measurements support the change for the uniform H1024 burst. The earlier four-pair comparison and these six rotations had different control rates. Timing other horizons and offered loads would require their own matched replays.

## Workload and measurement

Each W4 replay served 100 two-series AssetOpsBench MetroPT requests with context 512 and horizon 1024, offered 1 ms apart. All 100 reuse the same pair of sensor windows with distinct request and item IDs. We used the archived arrival plan, pinned TimesFM 2.5 checkpoint and independent native CUDA references for all nine quantiles. All three modes used the same A100-SXM4-40GB, Torch 2.14.0+cu130, vLLM 0.33.0 V1, eager FP32, TF32 disabled, four request slots, a 16-row execution cap and a 128-request queue. Both continuous variants used one-step prefill grouping.

Every arm started a fresh server and warmed it with three W4 requests before timing. Throughput divides total successful completions by the sum of the replay wall times. The latency percentiles pool each request's time from offered arrival through decoded HTTP response, including queueing. NVML measured peak device residency, and the worker reported Torch peak allocated and reserved memory. Numerical checks and diagnostic traces ran outside the timed intervals.

Each pair or three-way rotation used identical request IDs, payloads and arrivals. These are repeated bursts on one GPU. Differences between the two campaigns reflect run-to-run variation, so each comparison uses controls from its own campaign.

## Quantile assembly cost

A separate Nsight Systems capture for each continuous mode ran the full W4 replay. Both made 26 prefill and 176 autoregressive calls for 100 completed forecasts. The figures below combine the prefill and decode portions of `_finish_block`, amortized per completed forecast.

| W4 profile | #82 continuous | With #79 |
| --- | ---: | ---: |
| Kernel launches for output assembly | 160 | 26 |
| Inclusive host time in output assembly | 5.14 ms | 1.01 ms |
| Summed GPU time in assembly kernels | 0.664 ms | 0.109 ms |

A synthetic fixed-cohort check ran the same four two-series owners through eight forecast blocks. With eight warmups and 60 timed samples per implementation, median synchronized assembly time was **2.93 → 0.54 ms per forecast**. The two implementations produced bitwise-identical tensors. Peak extra Torch allocation across storage and assembly rose from 748,544 to 908,288 bytes in this probe. The full-horizon calculation uses larger temporaries at completion. The HTTP throughput measurements come from separate untraced runs.

## Admission-limit memory and request ownership

At the admission ceiling of C8192/H1024, we reset the Torch allocator peaks after clearing its cache and sampled whole-device NVML residency every 20 ms. With four active four-series requests, peak reservation and sampled GPU residency each fell 2 MiB. With successive 16-series requests, both peaks matched. All cases retained the same 1,861,222,400 K/V bytes.

| Maximum-size case | Mode | Peak Torch allocated | Peak Torch reserved | Peak device residency |
| --- | --- | ---: | ---: | ---: |
| Four requests sharing execution | #82 | 5,081,903,616 B | 5,402,263,552 B | 6,505,955,328 B |
| Four requests sharing execution | #79 | 5,081,756,160 B | 5,400,166,400 B | 6,503,858,176 B |
| Four 16-series requests admitted one at a time | Both | 3,934,203,904 B | 4,458,545,152 B | 5,562,236,928 B |

All 22 pinned-checkpoint stepping tests passed. An additional ownership test held H96 and H129 forecasts while canceling a different H1024 request and completing its H1024 peer. The requests exercised ragged histories, a partial block and cache positions 2 and 6 in the same decode. All 14 watched allocations belonging to the canceled request became reclaimable. The three completed forecasts occupied exactly 3,456, 9,288 and 73,728 bytes, respectively, and matched all nine native quantiles with maximum absolute error `3.46e-6`. The real HTTP suite also passed six staggered completion/disconnect and slot-reuse cycles, plus its overload and recovery checks.

Six HTTP integration cases passed against the independent native checkpoint. They included requested subsets `[0.2, 0.8]` and `[0.1, 0.5, 0.9]`, all nine quantiles and two pinned sensor cases. All 20 full-forecast graph tests passed, including dense and paged replays with changed inputs.

## Reproduction and raw evidence

The archive contains the full Git history needed for both revisions, the pinned workload, native references, per-request latency and error records, the Nsight captures and SQLite exports, maximum-memory probes, test logs and GPU cleanup evidence.

- Team A100 archive: `hpml:~/vtsfm-issue79-output-20261120/issue79-timesfm-output-ed5e9f28-a100.tar.gz`
- SHA-256: `0bc169e131bc232ce03c8fd38a1c51256f79cf53a1b9e22fc9efab8472af4dd3`
- W4 arrival-plan SHA-256: `3a7f93c57fc5d2a34b86961afe10a8df3a6dde68f69877f82e4ff5d917e74c8c`
- [Performance tracker](https://github.com/vinli0921/vTSFM/issues/51) and [#79 acceptance criteria](https://github.com/vinli0921/vTSFM/issues/79)

On the team host, use an idle GPU and a fresh results directory. The archived roots file identifies the two clean checkouts. To repeat the six untraced rotations without overwriting the evidence:

```bash
run="$HOME/vtsfm-issue79-output-20261120"
prior="$HOME/vtsfm-issue78-20261120"
py="$HOME/vtsfm-upgrade-spike/envs/timesfm25/bin/python"
replay=$(mktemp -d "$run/repeat-XXXXXX")
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0 \
  "$py" "$run/campaign.py" --prior "$prior" --experiment "$replay" --roots "$run/roots.json"
```
