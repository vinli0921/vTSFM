# Where the vLLM 0.33 forecast latency changed

Toto’s speedup and Moirai’s slowdown both appeared when the models ran directly, with vLLM out of the call path. The upgrade also changed the pooling scheduler default and the JSON response encoder. We measured those two changes independently. The model-specific adapter source, checkpoint weights and inputs were held fixed, but the compatible Torch, CUDA and supporting packages moved together.

## Comparing the serving paths

The table gives the change from 0.16 to 0.33 as **throughput % / p95 latency %**. For a direct call, throughput is the reciprocal of its mean synchronized wall time. Each direct and native HTTP result is the geometric mean of two opposite-order pairs. The vTSFM HTTP column uses the three default-mode pairs from [#33](https://github.com/vinli0921/vTSFM/issues/33). K is the number of concurrent HTTP clients.

| Model and workload | Direct native | Direct adapter | Native HTTP K1 | Native HTTP K4 | vTSFM HTTP K1 | vTSFM HTTP K4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toto · 128 context · 4 series | +3.7 / −3.0 | +2.9 / −1.2 | +2.2 / +1.0 | +4.5 / −4.6 | +3.7 / −4.4 | +4.6 / −6.0 |
| Moirai · 512 context · 1 series | −4.5 / +4.6 | −4.4 / +4.9 | −2.6 / +2.4 | −3.7 / +5.6 | −2.2 / +0.8 | −1.2 / +2.0 |
| TimesFM · 512 context · 4 series | −2.3 / +4.8 | −2.1 / +2.6 | −2.6 / +2.7 | −2.7 / +1.4 | +2.1 / −5.1 | +0.1 / −2.4 |

- **Toto.** In both orders the direct native mean fell by 3.7–4.8 ms. The speedup survived the frozen adapter and the separate native HTTP server at K4. At K1, native HTTP p95 changed direction across trials. The historical vTSFM gain at K4 includes a faster underlying forecast path as well as serving changes.
- **Moirai.** Direct native throughput fell 2.5% and 6.5% in the two trials. Native HTTP p95 at K4 rose 5.7% and 5.6%. The slowdown is measurable before vLLM schedules a request. In the matched synchronous trace, the model forward took 3.5 ms longer and the mean submission-to-scheduling interval grew by 12.0 ms with four clients and one slot. This is consistent with a slower model call making other requests wait longer.
- **TimesFM.** Direct native inference slowed in both orders, while default vTSFM HTTP at K4 changed direction across the three #33 trials. The trace and timings describe this workload; they do not establish a general TimesFM HTTP regression.

The [untraced measurements](timings.csv) give the sample count, absolute latency, throughput, request hash and trial order for each comparison.

## What changing the scheduler explains

All six frozen startup configurations confirmed the default switched from asynchronous in 0.16 to synchronous in 0.33 for pooling. Both modes worked on both versions. On this single-GPU setup, async permits two outstanding engine batches and sync permits one. The forecast request-slot limit remained one throughout.

Four-client p95 in milliseconds, with one fresh server per cell and 100 requests (200 for Moirai). The second sweep reversed the four configuration launches for each model. Full K1 and K4 absolute rates and p50/p95 are in the [scheduler measurements](scheduler.csv).

| Model | Sweep | 0.16 async (default) | 0.16 sync | 0.33 async | 0.33 sync (default) |
| --- | --- | ---: | ---: | ---: | ---: |
| Toto | 1 | 593.0 | 483.5 | 494.5 | 477.7 |
| Toto | 2 | 479.8 | 509.6 | 471.9 | 468.0 |
| Moirai | 1 | 154.3 | 151.8 | 174.5 | 164.1 |
| Moirai | 2 | 148.4 | 140.8 | 157.7 | 149.5 |
| TimesFM | 1 | 430.4 | 441.5 | 447.1 | 428.7 |

- **Toto.** Forcing 0.33 back to async raised p95 by 16.8 ms in the first sweep and 3.8 ms in the reverse sweep. At fixed sync, the new release still reduced p95 by 1.2% and 8.2%. The 593.0 ms old-async reading was about 480 ms on repeat. The 0.16 toggle changed sign, so a single percentage of the historical gain attributable to scheduling is unsupported.
- **Moirai.** Holding the release fixed at 0.33, sync reduced p95 by 5.2–6.0% relative to async. Holding both releases at sync, 0.33 still increased p95 by 6.2–8.1%. The new default cushioned the observed slowdown.
- **TimesFM.** The one explicit sweep at K4 favored sync on the new release. The default cross-release difference in #33 varied by trial. This single sweep cannot establish a stable scheduler share.

## Inside a forecast request

Both V1 releases used **explicit synchronous scheduling** for these traces. The stage columns are means in milliseconds per request. The first four columns come from a diagnostic CPU window with 24 requests at K4. The last two come from a separate eight-request CUDA capture. GPU busy time is the union of kernel and copy/memset intervals. The CPU windows ran under an Nsight-launched server with CUDA collection paused. Their absolute durations can differ from uninstrumented runs.

| Four-client workload | Submission → scheduling | Worker input preparation¹ | Model forward | Response serialization² | GPU busy | Kernels / request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toto | 383.96 → 384.22 | 0.53 → 0.92 | 140.67 → 139.61 | 2.32 → 0.58 | 15.85 → 16.01 | 3214 → 3268 |
| Moirai | 114.13 → 126.10 | 0.50 → 0.93 | 41.75 → 45.22 | 0.66 → 0.14 | 6.66 → 6.70 | 1269 → 1277 |
| TimesFM | 363.36 → 385.70 | 0.51 → 0.88 | 133.93 → 140.70 | 2.38 → 0.54 | 29.62 → 29.75 | 6058 → 6144 |

¹ The full stage table separates parsing, frontend staging, numerical transport, worker input preparation and pooling. Submission-to-scheduling joins frontend submission, scheduler admission, and the first scheduling decision for the same request ID. It includes IPC and waiting behind admitted work. The actual scheduler function time is reported separately.

² On 0.16 this is the response model dump plus Starlette JSON rendering. On 0.33 it is the inclusive response-creation function, which includes the dump. The encoder-only test below isolates the time spent converting the Python response object into bytes.

GPU busy time at K4 moved by at most 0.16 ms per request across these three captures. Moirai copied 8,988 bytes per request from device to host on both releases. These traces show no GPU-work or transfer-volume change large enough to account for the measured Moirai queue growth. Transfer counts alone cannot rule out a host-side wait. These captures do not distinguish Python, Torch host dispatch or an individual dependency as the cause of its longer forward. See the [CPU spans](cpu-stages.csv) and [CUDA work, transfers and host sync](cuda-activity.csv).

## How much JSON encoding affected Toto

The 0.16 pooling route renders the response with the standard-library JSON encoder. The 0.33 route uses `orjson`. On the same interpreter and the exact forecast values, six paired encoder-only trials reduced median encoding time from **2.08 to 0.12 ms** for Toto, **0.53 to 0.03 ms** for Moirai and **2.09 to 0.12 ms** for TimesFM. Decoding either output gave the same values.

We then changed only the encoder inside the 0.33 Toto response function, keeping the model dump, response envelope, synchronous scheduler and model fixed. Each of the two opposite-order pairs used fresh servers and 100 timed requests per cell. Switching from standard-library encoding to the stock encoder gave these results.

| Clients | Throughput change, pair 1 / pair 2 | Mean latency change, ms | p95 change, pair 1 / pair 2 |
| --- | ---: | ---: | ---: |
| 1 | +1.78% / +0.84% | −2.17 / −1.03 | −1.67% / −0.64% |
| 4 | +0.05% / +0.70% | +0.01 / −3.26 | +0.77% / −4.33% |

The encoder recovered 1–2 ms for a single client in both pairs. With four clients, the throughput benefit was smaller than #33’s 3.8–5.4% whole-stack gain and the p95 effect changed direction. A faster encoder is one cause of the single-client improvement. Different measurement boundaries and independent runs prevent assigning additive shares of the four-client gain to inference, scheduling and serialization. [Encoder HTTP measurements](encoder-http.csv)

## Measurement boundaries

Each point uses the original AssetOpsBench MetroPT request body at horizon 96 and all nine quantiles. The context and series count per HTTP request are 128/4 for Toto, 512/1 for Moirai, and 512/4 for TimesFM. One or four clients issue requests to a single model slot. Four clients can wait for work already ahead of them, so their p95 includes both service time and queueing.

| Path | Start and stop |
| --- | --- |
| Direct native | Begin with host histories, call the upstream model, return the quantile array to the host and synchronize the device. |
| Direct adapter | Begin with the same host histories, parse and stage them, call the frozen adapter with the same weights, return the quantile array to the host and synchronize. |
| Native HTTP | Send the pre-encoded request to a single-slot native server and decode the JSON response. |
| vTSFM HTTP | Send the same pre-encoded request to the vLLM V1 pooling server and decode the JSON response. |

The control is [vTSFM 65bd0c8](https://github.com/vinli0921/vTSFM/commit/65bd0c8171754c920da36094439fad8f6cc6a490) with vLLM 0.16.0, Torch 2.9.1, CUDA 12.8 and Transformers 4.57.6. The candidate is [f122dda](https://github.com/vinli0921/vTSFM/commit/f122ddab88cdd26ece4446b18ddcf4fcbb137d28) with vLLM 0.33.0, Torch 2.14.0, CUDA 13.0 and Transformers 5.20.2. The three model-specific runtime source files were identical across commits. The releases also share checkpoint revisions and upstream model-package versions (TimesFM 3.0.2, Toto 2.0.0, uni2ts 2.0.0). vLLM, the vTSFM integration code, Torch, CUDA, Transformers and several supporting libraries changed together.

The runs used the same A100-SXM4-40GB as the three trials in #33, eager FP32, TF32 disabled, and one vLLM scheduler slot. Startup, warmup, parity checks and traces were excluded from baseline timing. Throughput is successful completions divided by elapsed wall time. Queued HTTP clients contribute to a request’s latency. Direct path durations, CPU function spans and CUDA kernel activity overlap or have different boundaries. They should not be added together or subtracted from a p95 to estimate exclusive host cost.

## Traces and verification

Download an archive, extract it, decompress a JSON trace and open it in a Perfetto or Chrome trace viewer. Each archive includes both releases, the K1 and K4 request timelines and the corresponding CUDA capture with kernel, copy and host API events. Request numbers and timestamps are local to each exported trace. [Toto traces](https://github.com/vinli0921/vTSFM/releases/download/profiling-frozen-runtime-a100-issue34-2026-11-21/toto2-timelines.zip) · [Moirai traces](https://github.com/vinli0921/vTSFM/releases/download/profiling-frozen-runtime-a100-issue34-2026-11-21/moirai2-timelines.zip) · [TimesFM traces](https://github.com/vinli0921/vTSFM/releases/download/profiling-frozen-runtime-a100-issue34-2026-11-21/timesfm25-timelines.zip)

All **10,000 untraced timed operations** succeeded: 5,200 native/adapter calls and native HTTP requests, 4,000 vTSFM scheduler requests and 800 Toto encoder HTTP requests. Separate parity checks used the archived #33 forecasts; maximum absolute difference was `1.91e-6` at `atol=rtol=1e-4`. All 288 diagnostic HTTP responses matched their own release’s archived forecast exactly. Accepted runs passed GPU process-ownership and cleanup checks. Runs interrupted by another workload or with ambiguous process ownership were excluded.

The team archive includes original Nsight reports, SQLite exports, raw timing samples, rejected-attempt records, manifests, workload hashes, and scripts for offline reanalysis. Its location is `hpml:~/vtsfm-issue34-20261121/frozen-attribution-a100-20261121.tar.gz` and its SHA-256 is `afc988c8dbe548fb8fc3220ca649ce3715ea54b5d41297c1e7b1c222e4f94805`. See `REPRODUCE.md` inside the archive. Two direct pairs, one or two scheduler sweeps per model and one trace per release and model support the comparisons above. No binary-compatible one-library-at-a-time experiment was performed, so the underlying native timing changes cannot be assigned individually to Torch, CUDA or Transformers.
