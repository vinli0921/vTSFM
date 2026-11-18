# Complete TimesFM forecast graphs — November 13, 2026

These results describe the measured revisions and their separate sign prefills.
The current executor batches both signs at prefill. Rerun the four-arm
comparison before using these timings for the updated executor.

## Outcome

Full numerical forecast capture works for both backends. On the matched fixed
B2/C512/H1024 workload, dense graphs are **5.84× faster** than dense eager, and
paged graphs are **7.00× faster** than paged eager. Paged graph mean latency is
**7.01% lower** than dense graph latency. Graphs increase the sampled device
footprint by approximately 156–172 MiB relative to their eager controls.

This is a fixed-shape, serialized, in-process experiment. It supports progressing
to opt-in adapter/worker integration; it does not enable graphs in vLLM serving,
dynamic shape caches, continuous scheduling or HTTP cancellation. The preceding
[attention-only experiment](../2026-11-12-native-kv/combined.md) remains separate.

## Sources and matched execution

| Item | Setting |
| --- | --- |
| Measured source, clean checkout | `1c9377c91e2b708fb21b8b5ad495097a7ca17dfc` |
| Parent experiment / PR #63 | `59aaa69250369529e8432c074078f1dd09703c8c` |
| Shared optimized executor, PR #61 | Based on `58a3c843984c932f0eaf837a79fa496872260120` |
| TimesFM checkpoint | `google/timesfm-2.5-200m-pytorch`, `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| Upstream TimesFM | Package 3.0.2, `e51928e27119cb17bebc005be2696b75e0a9e688` |
| GPU | A100-SXM4-40GB; driver 595.91.07 |
| Runtime | Python 3.11.17, vLLM 0.33.0, Torch 2.14.0+cu130/CUDA 13.0, Triton 3.8.0 |
| Precision | FP32, TF32 disabled, `TRITON_F32_DEFAULT=ieee`, four Torch threads |

All four arms use `FixedForecast.run` and the same graph-safe numerical body.
The graph contains outer normalization, both sign prefills, all eight forecast
blocks and running-stat updates, both heads, quantile correction, denormalization
and positivity handling. Prefill uses the shared explicit helper with known
positions, removing the upstream CUDA-scalar slice readback. Default serving
retains its original prefill call through a small forwarding hook.

Finite-input validation, static CUDA input staging and an independently owned
output clone remain outside the captured graph and inside every measured call.
The eager controls use the same wrapper. All arms fix shape/horizon; paged plans
and leases are prepared at setup in both eager and graph modes. Dense masks and
positions are computed inside the numerical body and included in its graph.
GPU counters/statistics reset on every
replay; paged KV is overwritten before use. Pages remain owned until graph reset
and synchronized teardown on the arena's owning stream.

These controls differ from the earlier lifecycle-inclusive KV comparison, which
allocated/released leases per forecast and used a different prefill path. Use
the new eager controls to attribute speedups to graph execution; do not compute
an isolated graph speedup from the earlier report's eager numbers.

## Workload and method

The cached AssetOpsBench MetroPT fixture uses observed sensor values and its
documented causal forward-fill. Data source/revision and preparation are unchanged
from the [original provenance](../2026-11-12-native-kv/README.md#sources-and-method).
AssetOpsBench source is `e844f6468c6c68b328022e7ca8e31da140842cc7`;
prepared `data.npz` SHA256 is
`b6f80b8e9b7e11eaca0d3c5c61b2a4f3e8d290259653e3fe47870dc45b7e1b6d`.
Two independent model rows use C512 histories and H1024 forecasts, returning all
nine quantiles. Their input SHA256 is
`db459fe68b795882f093e47e2679155b220e12e528e543e7ce4c24f87b65f596`.
Rows/windows are correlated observations from one compressor, not independent
assets. A synthetic transform B = `-A*1.7 + 0.3` is used only for replay parity;
no observed-label forecast-quality claim is made.

Each of four rounds rotates all four arms through every order position, with
fresh processes: 16 children, 100 timed forecasts each, **400 per arm / 1,600
total**. Construction has three body warmups; parity is followed by five wrapper
warmups. Setup/capture, native references, input preparation, ownership queries,
profiling and destruction are outside timing. Timing includes `run` and CUDA
synchronization. Forecasts/s is reciprocal mean latency for one sequential B2
forecast, not saturated or HTTP throughput. P95 pools the 400 raw samples with
linear interpolation. Four-round ranges are not confidence intervals.

## Latency and reciprocal throughput

| Arm | Mean ms | Pooled p95 ms | Round-mean range ms | Sequential B2 forecasts/s |
| --- | ---: | ---: | ---: | ---: |
| Dense eager | 365.967 | 387.554 | 361.067–373.496 | 2.732 |
| Paged eager | 407.767 | 417.021 | 404.948–410.299 | 2.452 |
| Dense graph | 62.690 | 62.838 | 62.646–62.736 | 15.952 |
| Paged graph | 58.293 | 58.410 | 58.280–58.308 | 17.155 |

Within-backend mean speedups are 5.8378× dense and 6.9951× paged. Paged graph
latency is 7.0124% lower than dense graph, with the same ordering in every round.
Paging remains slower in eager execution. This workload therefore demonstrates
a full-forecast benefit from combining paging with graph execution, rather than
only the earlier single-attention replay result. It does not establish the
ordering for other batch sizes, contexts, horizons or GPU architectures.

## Setup and memory costs

| Arm | Total owner setup range, s | Warmup range, s | Capture range, s |
| --- | ---: | ---: | ---: |
| Dense eager | 1.085–1.124 | 1.078–1.117 | 0 |
| Paged eager | 9.663–9.857 | 2.697–2.814 | 0 |
| Dense graph | 1.606–1.633 | 1.070–1.097 | 0.528–0.538 |
| Paged graph | 10.212–10.459 | 2.689–2.835 | 0.542–0.575 |

Total setup includes initialization/imports and warmup; model checkpoint loading
precedes it. The roughly ten-second paged setup is **not** graph capture time.
These costs are paid before steady-state timing and must be amortized over owner
reuse; reuse/eviction under a real workload is unmeasured.

Maximum counters across four rounds, MiB:

| Arm | Setup Torch allocated / reserved | Timed Torch allocated / reserved | Timed sampled device/NVML |
| --- | ---: | ---: | ---: |
| Dense eager | 937.470 / 1024 | 937.466 / 1024 | 2074.563 |
| Paged eager | 943.861 / 1028 | 943.857 / 1028 | 2078.563 |
| Dense graph | 937.470 / 1046 | 898.708 / 1046 | 2246.563 |
| Paged graph | 943.861 / 1050 | 939.352 / 1050 | 2234.563 |

After parity/warmup, GC/empty-cache run outside timing and Torch peaks reset.
Graph/private pools and fixed paged storage remain held. NVML samples at 20 ms
and may miss transients; setup NVML peaks were not measured. Lower graph-mode
`memory_allocated` is **not** a VRAM saving: addresses can remain retained in a
private pool after their Python tensors cease to count as active allocations.
Report reserved and device memory as well. Device peaks increase by 172 MiB
dense and 156 MiB paged; the study does not attribute that increment to one
allocator or runtime component.

Paged owners intentionally hold four independent sign leases: 37.5 MiB of
leased pages and a 40.625 MiB fixed pool, including unused/null pages. All leases
and physical pool bytes return to zero after close. Setup/warm/timed counters
and lease records are retained individually in the raw point files.

## Correctness and actual graph activity

- All 48 recorded A/B/A comparisons pass across all nine quantiles at unchanged
  `rtol=atol=1e-4`, maximum absolute error `9.5367431640625e-7`. Retained A and B
  outputs remain unchanged after later calls, detecting stale inputs and aliases.
- Every graph point has a real `CUDAGraph` and exactly 100 timed replays; eager
  points have zero. A separate four-arm trace shows one `cudaGraphLaunch` in each
  graph arm and zero in eager arms. The complete forecast is replayed; graph
  execution does not imply fewer numerical kernels.
- Clean measured source passes 12 complete-forecast graph checks covering
  C31/32/33/512/8192, horizon boundaries, row perturbation, invalid-input rejection
  and cleanup, plus 56 native/prefill/helper regressions. The CPU suite passes
  536 tests (19 optional-runtime skips). Ruff, package build and fresh installed
  wheel CPU-safe discovery pass.
- All 3,264 primary ownership snapshots contain no foreign PIDs or query errors.
  Snapshot checks are outside timing and cannot exclude very brief activity
  between checks. All 16 owned children stop without cleanup errors; the GPU was
  idle afterward. Inputs, package/source hashes, checkpoint, precision and GPU
  match across arms. Independent remote/local aggregation agrees.

## Interpretation and next work

Full-forecast capture is the major improvement in this study. The paged backend
adds a smaller measured gain after capture on this shape. Proceed with opt-in
TimesFM adapter/worker integration and a bounded graph-owner/shape policy under
#17, measuring real HTTP latency, admission memory and retained-worker reuse.
Validate additional shapes before selecting a default backend. Integration is
future work; this PR does not complete #17's serving acceptance.

One owner supports serialized finite equal-length inputs at fixed B/C/H. Dynamic
mask/shape changes, concurrent owner calls, cross-request cancellation and
per-block joins are unsupported. This is not native GPUModelRunner/EngineCore
integration, nor proof of graph behavior for Toto or Moirai. No new kernel is
claimed.

## Reproduce and artifacts

Use the [manual guide](../../timesfm-full-graph.md) at the measured source with
the pinned Linux TimesFM environment and cached checkpoint/data. Run the four
arms with `--requests 100 --warmups 5 --setup-warmups 3 --repetitions 4`.
Run traces separately with a new output directory and `--profile`; do not use
the two-sample traced smoke as statistical performance evidence.

Raw data, setup/cleanup logs, traces and the exact source bundle are retained
remotely under `~/vtsfm-fullgraph-20261113/`. The ignored local archive is
`results/fullgraph-20261113/evidence.tar.gz`, SHA256
`840140a412e9368d82900a57bcd7c8abe3ddaa9a8c60b5abb73a456214eac08d`.
`completion.json` records successful manifests and an idle GPU. Local
`aggregate_and_plot.py` validates 35 raw JSON hashes, exact counts, source/pins,
ownership, parity and cleanup, producing `aggregation.json` and the
`timesfm-full-forecast-graphs.png`/`.pdf` figures. Those artifacts remain outside
Git; the reviewed summary here is the published record.
