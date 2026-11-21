# TimesFM full-forecast CUDA graphs over vLLM HTTP — November 17, 2026

Full-forecast CUDA graphs delivered **4.69× (dense)** and **5.44× (paged)**
speedups in mean HTTP latency over the matched eager paths.
With both backends captured, paging lowered mean latency by **5.49%** and
p95 by **5.52%**. Paged eager took **9.64%** longer than dense eager.

| Mode | Mean ms | Pooled p95 ms | HTTP requests/s | Round mean range, ms |
| --- | ---: | ---: | ---: | ---: |
| Dense eager | 335.89 | 345.32 | 2.98 | 333.54–339.62 |
| Paged eager | 368.27 | 377.21 | 2.71 | 363.68–372.16 |
| Dense graph | 71.64 | 72.78 | 13.92 | 71.31–72.05 |
| Paged graph | 67.70 | 68.76 | 14.73 | 67.33–67.96 |

## Workload and method

The [TimesFM serving experiment](../../timesfm-graph-serving.md) uses vLLM V1's
`POST /pooling` endpoint on one A100-SXM4-40GB with driver 595.91.07. The
measured vTSFM checkout was clean at
`66de90ef18d3e4ae0fbbddde7920c6554a4a943f`. It includes the fused prefill
for TimesFM's normal and flipped rows from PR #64. It ran vLLM 0.33.0,
Torch 2.14.0+cu130, and upstream TimesFM 3.0.2 at
`e51928e27119cb17bebc005be2696b75e0a9e688` with checkpoint
`google/timesfm-2.5-200m-pytorch` at
`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. All arms ran in FP32 with
TF32 disabled and `TRITON_F32_DEFAULT=ieee`.

Each timed request contains two independent model rows with 512 history
values, a 1,024-point horizon, and nine requested quantiles. Inputs come from
the same two sensor windows used in the
[in-process graph study](../2026-11-13-full-forecast-graphs/README.md#workload-and-method)
(channels 0 and 1, starts 0 and 131). The windows use the prepared
AssetOpsBench MetroPT fixture at source revision
`e844f6468c6c68b328022e7ca8e31da140842cc7`, which includes causal forward
fill. The prepared NPZ SHA-256 is
`b6f80b8e9b7e11eaca0d3c5c61b2a4f3e8d290259653e3fe47870dc45b7e1b6d`.
The changed-input parity probe applies `-1.7 × A + 0.3`. Its input is
synthetic and used only to check replay correctness.

Three complete rounds each start a fresh server per mode, run five parity
probes and five HTTP warmups, and measure 100 requests on a persistent loopback
connection with one outstanding request. Each mode runs in three of the four
order positions. A foreign GPU process appeared during the fourth round's
paged-eager startup. The harness stopped its own processes. The entire fourth
round is excluded, including the two modes that finished before contention.

Latency starts at the pre-encoded POST and ends after JSON decoding and
response ID validation. P95 pools 300 raw latencies per mode with linear
interpolation. Throughput divides 300 completed requests by the sum of the
three timed wall intervals. Startup, parity, and warmups are measured
separately. The eager controls share the same forecast code, input staging,
and owned output copies as their corresponding graph modes. The fallback
probes exercise the ordinary TimesFM HTTP path.

This experiment measures fixed-shape, single-request serving on one GPU.
Multi-client serving, dynamic shape caching, and forecasting accuracy against
future observations remain outside its scope.

Run the same study with the
[manual validation commands](../../timesfm-graph-serving.md#validation-and-benchmark).

## Setup and GPU memory

| Mode | Server ready, s | Owner setup, s | Capture, s | Peak Torch allocated / reserved, MiB | Peak sampled NVML, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense eager | 27.06–27.07 | 1.20–1.25 | 0 | 941.85 / 1,028 | 2,078.56 |
| Paged eager | 28.57–29.58 | 2.75–2.78 | 0 | 948.11 / 1,028 | 2,080.56 |
| Dense graph | 27.56–28.07 | 1.68–1.73 | 0.46–0.50 | 899.02 / 1,050 | 2,230.56 |
| Paged graph | 29.07–29.57 | 3.25–3.28 | 0.47–0.49 | 939.67 / 1,050 | 2,220.56 |

Server readiness includes model loading and owner setup. Setup includes owner
warmups and capture. The memory columns are maxima across the three timed
rounds. NVML samples whole-device residency every 20 ms. Graph replay added
152 MiB for dense and 140 MiB for paged by that measure. Both paged modes
retained 37.5 MiB in leases inside a 40.625 MiB fixed KV pool.

## Correctness and execution

All 60 native parity checks passed on all nine quantiles at `rtol=atol=1e-4`,
with a maximum absolute difference of `9.54e-7`. The probes change input
values and request IDs, return to the initial values, run an eager shape
fallback, then return to the fixed shape. Each graph mode captured once per
server startup and replayed 300 times in the timed samples. The eager modes
each made 300 fixed-owner calls. Timed fallbacks and recaptures were zero.

## Disconnect and maximum-size fallback

A separate A100 lifecycle run at [`06b018a`](https://github.com/vinli0921/vTSFM/commit/06b018ab7e0e5a3d02a7ae5eb635a81f12d0a3d5)
exercised both graph modes after the merge with `main`. Each mode recovered
from a queued disconnect and a disconnect after the request entered the
worker. It then served one eager B16/C8192/H1024 fallback with the fixed owner
resident and replayed the original graph on the next matching request. Each
server recorded one capture and five replays across the test. Both servers
shut down and released the GPU.

| Mode | Peak Torch allocation during maximum-size fallback |
| --- | ---: |
| Dense graph | 3,760 MiB |
| Paged graph | 3,801 MiB |

The worker reset Torch's peak counters immediately before the fallback. The
steady-state HTTP table above measures B2/C512/H1024 on `66de90e`.

## Evidence

The raw archives contain per-request timings, parity probes and responses,
route counters, ownership checks, cleanup logs, and Git bundles for the
measured source. They are stored under the ignored
`results/timesfm-graph-http-20261117/` directory.

| Archive | SHA-256 |
| --- | --- |
| `http-four-arm-merged-evidence.tar.gz` | `3c45a85df58f1bb070fd9b54a2d19bd1b039a94812ed84182f215f7d700df6d9` |
| `http-four-arm-first-evidence.tar.gz` | `a3f14a5b688cc65fde235206c4a3d2f9b918732be26779e63fecf4877ecaf99e` |
| `lifecycle-pr70-06b018a-evidence.tar.gz` | `9de555684ac9bbaf67d3f3d75c9de4d796d65453e32efbb5fac786783a99193a` |

## Earlier serving revision

The same protocol ran on clean `2f1394456ab6d86dd88af3a7d2b87a7f1595b189`
before the merged sign-prefill changes. Those 1,600 requests form a separate
baseline. The earlier lifecycle check measured peak Torch allocation of
2,491 MiB for dense and 2,532 MiB for paged during maximum-size fallback,
before batched sign prefill. Several merged changes affect timing, including
prefill and HTTP input parsing.

| Mode | Mean ms | Pooled p95 ms | HTTP requests/s |
| --- | ---: | ---: | ---: |
| Dense eager | 377.95 | 386.66 | 2.64 |
| Paged eager | 413.59 | 422.38 | 2.42 |
| Dense graph | 79.75 | 80.88 | 12.51 |
| Paged graph | 74.97 | 76.02 | 13.30 |
