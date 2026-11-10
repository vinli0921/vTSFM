# Nsight Systems breakdown of per-request host overhead — November 10, 2026

Issue #16. Tools and reproduction: [benchmarks/nsight](../../../benchmarks/nsight/README.md).

A100-SXM4-40GB (shared GCP VM), driver 595.91.07. vLLM 0.33.0 V1, Torch 2.14.0+cu130, eager FP32, TF32 off,
async scheduling off. vTSFM `origin/main` 7fa1c5c. Nsight Systems 2026.5.2 (user-space install).
Data and rounds follow `profiling/harness.py` (branch `profiling-baseline`, mirrored in `benchmarks/nsight/workload.py`): chiller_6 supply temperature,
context 2048, horizon 96, all 9 quantiles. Round 1 = batch 1, window advances one patch per request (50 timed
requests). Round 0b = batch 64, fixed windows (30 timed requests). Five warm-up requests before each.

## Answer

- **At batch 1 all three models are host-bound.** The GPU is busy for 16–27% of each request. Most of the rest
  is eager PyTorch op dispatch (Python + ATen) plus `cudaLaunchKernel`. Sync waits and data conversion are small.
  TimesFM launches about **11,000 kernels per forecast**, Toto about 3,200 and Moirai about 1,200.
- **The vLLM/HTTP path adds little at batch 1:** 14 ms for TimesFM, 9 ms for Toto and 10 ms for Moirai. For
  Moirai that is 22% of the request; for TimesFM it is 8%.
- **At batch 64 the in-process paths are GPU-bound**: 78–92% GPU busy. Over HTTP, the frontend becomes the
  biggest host sink: `parse_request` validates each float in Python, which costs about **80–87 ms per
  64×2048 request**.
- **Block mode does pay a large fixed cost per model call.** Each forecast-block call is a full eager decode of
  about 5,000 kernel launches, roughly 100 ms of host time, whatever its row count. Requests at different cache
  positions cannot share a call. So continuous mode makes about **1.9× more launches per request** than whole
  mode (21.0k vs 11.0k), and also copies about 359 MB of caches device-to-device per request. With the GPU only
  14% busy, this explains the throughput gap: 2.29 vs 4.81 req/s.
- **Which fix first: CUDA graphs / torch.compile (#17).** It targets the top sink for every model and for block
  mode. Second is a thin request path (vectorized input validation and tensor build), which matters only for
  large-batch HTTP requests. vLLM scheduler cost is negligible (`Scheduler.schedule` takes about 0.2 ms per
  step).

## Method and limits

- Three paths per model. **native** is the official upstream API (TimesFM `forecast()`,
  `Toto2Model.forecast`, uni2ts `Moirai2Forecast.forward`). **runtime** calls the vTSFM adapter's
  `runtime.forecast` in-process, as `ForecastModel.forward` does. **HTTP** sends POST /pooling to the vLLM
  server with one request outstanding. Block mode is the saturated case from `benchmarks/continuous.py`:
  B2/C512/H1024, 50 requests offered 1 ms apart, 4 slots, fusion cap 8, with continuous and whole on the same
  load.
- Trace options: `--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=node`. In-process runs capture only
  the timed requests (`cudaProfilerStart/Stop`). The server runs under `nsys launch`, with `nsys start/stop`
  around the timed requests.
- **No CPU, Python or context-switch sampling.** `kernel.perf_event_paranoid=3`, and lowering it needs sudo.
  Instead, about 40 request-path functions in the API server and EngineCore are wrapped in NVTX ranges by a
  `sitecustomize` hook (`VTSFM_NVTX=1`). No source changes. "Python/framework" is the request time on the
  compute thread minus its CUDA API time.
- **Tracing overhead.** At batch 1, CUDA tracing adds about 6–8 µs to each launch, which inflates wall time
  1.5–1.9×. At batch 64 it adds 2–5%. Every config also has an untraced timing run. The host split below scales
  the traced non-sync host time to the untraced wall time. A light trace (`cuda,nvtx` only) gives the same
  shares.
- **TimesFM's admission policy caps a request at 16 rows** ([`admission.py`](../../../src/vtsfm/adapters/timesfm25/admission.py)). A 64-row HTTP
  request gets HTTP 400, so TimesFM's batched HTTP case uses sequential 16-row requests.
- Chronos-2 was not profiled. It uses a separate legacy environment, which the Nov 9 update keeps out of this
  stack.

## Per-request split, round 1 (batch 1)

Untraced wall times. The host columns are the traced split scaled to untraced wall time. GPU busy is the union
of kernel and copy time in each request.

| Model / path | Wall ms | GPU busy ms (%) | Python+framework ms | Launch ms | Sync wait ms | Copy API ms | Kernels / req | Host µs / kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimesFM native | 185.3 | 42.7 (23%) | 131.5 | 48.7 | 1.7 | 1.5 | 10,995 | 16.7 |
| TimesFM runtime | 164.9 | 42.7 (26%) | 115.0 | 45.0 | 1.7 | 1.4 | 10,958 | 14.9 |
| TimesFM HTTP | 179.2 | 42.7 (24%) | 125.9 | 48.2 | 1.8 | 1.6 | 10,969 | 16.2 |
| Toto native | 127.7 | 19.2 (15%) | 102.9 | 23.2 | 0.0 | 0.4 | 3,208 | 39.8 |
| Toto runtime | 118.1 | 19.2 (16%) | 94.7 | 21.9 | 0.0 | 0.4 | 3,212 | 36.8 |
| Toto HTTP | 126.7 | 19.2 (15%) | 102.3 | 22.7 | 0.1 | 0.5 | 3,222 | 39.3 |
| Moirai native | 36.9 | 9.9 (27%) | 28.1 | 7.5 | 0.4 | 0.6 | 1,228 | 29.8 |
| Moirai runtime | 36.4 | 9.7 (27%) | 27.8 | 7.1 | 0.4 | 0.6 | 1,234 | 29.1 |
| Moirai HTTP | 46.5 | 10.0 (22%) | 36.5 | 8.2 | 0.5 | 0.8 | 1,244 | 37.0 |

Mean traced launch cost is 7.7–10.6 µs per `cudaLaunchKernel`, including tracing overhead; the light trace
gives 7.3–9.8 µs. TimesFM issues 161 small device-to-host copies per forecast, each followed by a
`cudaStreamSynchronize`; Moirai issues 25–28. They cost under 3 ms because the GPU is nearly idle when they
happen. Most kernels are small elementwise ops: TimesFM and Toto run 2–4 µs elementwise kernels tens of
thousands of times per 50 requests.

**HTTP path at batch 1** (traced NVTX means per request; the overhead is roughly the same for all three
models):

| Part | ms |
| --- | ---: |
| Frontend `_preprocessing` (incl. `parse_request` 1.3, `pre_process` 0.4) | 2.3 |
| EngineCore input queue / IPC (`_process_input_queue`) | 5.2–5.5 |
| `GPUModelRunner.execute_model` around the model call (`_prepare_inputs`, `_pool`, state) | 2.5–3.0 |
| Frontend `_postprocessing` | 0.5 |
| `Scheduler.schedule` + `update_from_output` | 0.3 |

## Round 0b (batch 64)

| Model / path | Wall ms (untraced) | GPU busy ms (%) | Main host items |
| --- | ---: | ---: | --- |
| TimesFM native | 362.3 | 286.7 (79%) | 10.8k launches (~63 ms) overlap GPU; 108 ms sync wait at the end |
| TimesFM runtime | 367.8 | 286.7 (78%) | same as native |
| TimesFM HTTP, 16 rows | 225.4 | 100.2 (45%) | same 10.8k launches as batch 1; `parse_request` 19.9 ms |
| Toto native | 201.3 | 196.3 (98%) | GPU-bound; host fully overlapped |
| Toto runtime | 213.1 | 196.2 (92%) | GPU-bound |
| Toto HTTP | 387.7 | 196.3 (51%) | `parse_request` 87 ms, `pre_process` 10, `post_process` 8; about 60 ms outside the handler (JSON body, 1.4 MB) |
| Moirai native | 219.7 | 210.5 (96%) | GPU-bound |
| Moirai runtime | 244.4 | 212.1 (87%) | +25 ms vs native: `predict()` NumPy round trip and 510 extra kernels ([`moirai2/runtime.py`](../../../src/vtsfm/adapters/moirai2/runtime.py)) |
| Moirai HTTP | 410.5 | 212.1 (52%) | `parse_request` 80 ms, `pre_process` 10, `post_process` 8; about 50 ms outside the handler |

At batch 64 the HTTP path adds about 170 ms per request for Toto and Moirai, almost all of it per-element Python
work on the inputs and outputs. In the HTTP traces `pack_result`/`_pool` wait for the GPU to drain. That is GPU
time showing up as a sync, not host work.

## Block mode (TimesFM, saturated B2/C512/H1024, 50 requests)

| | Whole | Continuous |
| --- | ---: | ---: |
| Throughput, untraced (req/s) | 4.81 | 2.29 |
| p50 / p95 latency, untraced (s) | 5.58 / 9.60 | 11.81 / 20.29 |
| Engine-step host time per completed request (traced, ms) | 360 | 745 |
| Kernel launches per request | 11,012 | 20,996 |
| Model calls per request | 0.28 (about 3.6 requests share each whole-forecast call) | 4.16 (0.52 prefill + 3.64 advance) |
| Kernels per model call | about 39,000 (8 blocks) | about 5,000 (1 block) |
| Requests sharing a call, per 128-step block | about 3.6 | about 1.9 |
| GPU busy per request (ms) / share | 56 / 15.6% | 103 / 13.8% |
| Device-to-device copies per request | 92 (0.08 MB) | 1,832 (359 MB) |
| `TimesFMStepper._store` + `_merge_caches` per request (traced, ms) | — | 57 |

Every block call pays the same eager decode, about 5,000 launches, even when it carries only one or two
requests. Calls only fuse requests at the same cache position. So continuous mode spreads the fixed cost over
fewer requests than a whole-forecast call does. The host work per request roughly doubles, matching the 2.1×
throughput loss. The GPU stays mostly idle. The cache merge/store copies add about 8% more host time and about
5 ms of GPU copies per request.

## Top 3 host-time sinks and the fix each points to

| Model | 1 | 2 | 3 | Points to |
| --- | --- | --- | --- | --- |
| TimesFM | Eager op dispatch in the decoder: ~115 ms of 165 at batch 1, ~11k ops per forecast | `cudaLaunchKernel`: ~45 ms (11k launches) | Serving path (+14 ms at batch 1); `parse_request` at 16 rows (20 ms) | CUDA graphs / torch.compile (#17); in block mode also fusing across cache positions and avoiding cache copies |
| Toto | Eager op dispatch: ~95 ms of 118 at batch 1 (~37 µs host per kernel, highest of the three) | `cudaLaunchKernel`: ~22 ms (3.2k launches) | HTTP input validation at batch 64: `parse_request` 87 ms + `pre_process`/`post_process` 18 ms | CUDA graphs / torch.compile (#17); thin request path for batched HTTP |
| Moirai | Eager op dispatch: ~28 ms of 36 at batch 1 | Serving path: +10 ms at batch 1 (22% of the request); `parse_request` 80 ms at batch 64 | `cudaLaunchKernel`: ~7 ms (1.2k); runtime `predict()` NumPy round trip, +25 ms at batch 64 | CUDA graphs / torch.compile (#17); thin request path; tensor-input runtime instead of `predict()` |

## Files and reproduction

- `summary.csv`: one row per trace, with the untraced timing and the traced per-request split.
- The traces (26 `.nsys-rep` files, about 440 MB gzipped), `nsys stats` CSVs and per-run JSON are not
  committed. They are kept outside the repository, on the profiling VM (`~/issue16`) and in the author's local
  results folder. They can be published as a GitHub release if wanted.
- Reproduce with `benchmarks/nsight/run.sh OUT_DIR timesfm25 toto2 moirai2` (set `NSYS`, `AOB_ROOT`,
  `HF_HOME`, and `METROPT_DATA` for the block-mode case). One ordered pass per configuration, without
  repeats or confidence intervals.
