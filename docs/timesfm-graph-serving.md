# TimesFM full-forecast graphs over HTTP

The experimental `--timesfm-forecast-mode` option selects a dense or paged
forecast inside the TimesFM vLLM V1 pooling worker. Use the pinned Linux
TimesFM environment and a compatible NVIDIA GPU.

```bash
./scripts/serve.sh --adapter timesfm25 --timesfm-forecast-mode dense-graph --port 8001
```

Select `paged-graph` for the paged backend. The corresponding `dense-eager`
and `paged-eager` modes use the same fixed forecast owner and numerical body
with ordinary eager execution. The `POST /pooling` request and response schema
is the standard TimesFM schema. A request with two finite, 512-value histories
and a 1,024-point horizon qualifies, regardless of the requested quantile subset.
Other shapes, including ragged histories, use the ordinary TimesFM forecast path.

One V1 worker on one GPU handles one scheduled request at a time in this mode.
Forecast-block scheduling and cross-request fusion require separate launch
configurations. The worker builds and warms one fixed owner after model loading
and before readiness. A startup check requires at least 512 MiB of device
headroom for owner setup. This is separate from numerical transport admission
and from vLLM's GPU memory utilization setting. Worker shutdown destroys the
graph before freeing its fixed buffers and paged leases.

The worker graph covers the complete numerical forecast. vLLM handles request
parsing, routing, scheduling, result packing and HTTP I/O. Inputs are validated
and copied into fixed buffers, and each replay returns a separately owned result
for the pooling copy stream. A disconnect releases the request through vLLM's
normal lifecycle. If the worker has already started that forecast, it can finish
the numerical work before serving the next request.

The experiment uses FP32, disables TF32, and sets Triton's IEEE precision for
paged modes. Use fixed weights throughout the worker lifetime. GPU resource
usage includes any retained graph pools and the paged KV pool, including while
handling eager fallback requests.

## Validation and benchmark

Use the pinned checkpoint and prepared AssetOpsBench fixture on an idle Linux
GPU. The harness starts a fresh loopback server for each mode in each round. It
checks nine-quantile outputs, item IDs, A→B→A requests and eager fallback against
an independent native reference. Worker counters verify the selected route and
the number of graph replays during the timed phase.

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 TRITON_F32_DEFAULT=ieee PYTHONPATH=src:tests:.
export VTSFM_RUN_TIMESFM_NATIVE_KV=1 VTSFM_RUN_TIMESFM_LIFECYCLE=1
.venv-timesfm25/bin/python -m pytest tests/test_timesfm_lifecycle.py -k graph_owner -q
.venv-timesfm25/bin/python -m benchmarks.timesfm_graph_http \
  --backend all --requests 100 --warmups 5 --repetitions 4 \
  --data-dir data --output results/timesfm-http-graphs-NEW
```

Latency runs from the pre-encoded HTTP POST through decoding the response JSON.
Each client keeps one connection and sends one request at a time. Start-to-ready
time, worker setup and first-request latency are saved separately. The harness
records Torch allocator counters and whole-device NVML memory throughout the
steady-state measurements.

The A100 lifecycle checks passed for dense and paged graphs. The tests
disconnect a queued request and another request that has entered the worker,
then verify the original owner still serves correctly. Both modes also handled
an eager B16/C8192/H1024 fallback while retaining the graph. Peak Torch
allocation during the fallback was 2,491 MiB for dense and 2,532 MiB for paged.
This is a fallback capacity check. The benchmark checks output parity against
the native reference on its own input cases.
