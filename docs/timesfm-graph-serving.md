# TimesFM full-forecast graphs over HTTP

The experimental `--timesfm-forecast-mode` option selects a dense or paged
forecast inside the TimesFM vLLM V1 pooling worker. Use the pinned Linux
TimesFM environment and a compatible NVIDIA GPU.

See the [A100 HTTP benchmark](baselines/2026-11-17-timesfm-graph-http/README.md)
for latency, throughput, GPU memory, and native forecast parity across all four
modes.

```bash
./scripts/serve.sh --adapter timesfm25 --timesfm-forecast-mode dense-graph --port 8001
```

Select `paged-graph` for the paged backend. The corresponding `dense-eager`
and `paged-eager` modes use the same fixed forecast owner and numerical body
with ordinary eager execution. The `POST /pooling` request and response schema
is the standard TimesFM schema. A request with two finite, 512-value histories
and a 1,024-point horizon qualifies, regardless of the requested quantile subset.
Other shapes, including ragged histories, use the ordinary TimesFM forecast path.

Use this mode for traffic dominated by B2/C512/H1024 requests. A single-GPU
V1 worker uses one scheduling slot and disables cross-request fusion, so a
burst of other shapes runs one eager forecast at a time. Launch a separate
configuration for cross-request fusion or forecast-block scheduling.

The worker builds and warms one fixed owner after model loading
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
paged modes. If the environment selects compilation or TF32, select
`--forecast-execution eager --forecast-precision fp32` for this experiment.
Use fixed weights throughout the worker lifetime. GPU resource
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

On the A100 at `06b018a`, dense and paged graphs both served a new request
after a queued disconnect and after a disconnect inside the worker. Each also
served an eager B16/C8192/H1024 fallback with its graph resident, then reused
the same graph. Peak Torch allocation during that fallback was **3,760 MiB**
for dense and **3,801 MiB** for paged. The [lifecycle results](baselines/2026-11-17-timesfm-graph-http/README.md#disconnect-and-maximum-size-fallback)
record the revision, replay counts and evidence archive.
