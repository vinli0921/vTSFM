# TimesFM complete-forecast CUDA graph experiment

See the [measured four-arm results](baselines/2026-11-13-full-forecast-graphs/README.md)
for source pins, setup/memory costs, validation and measurement limits.

`FixedForecast` in `adapters/timesfm25/full_graph.py` captures the complete
numerical forecast in one CUDA graph: outer normalization, both sign prefills,
all continuation blocks and running statistics, both heads, quantile correction,
denormalization and positivity handling. It uses the optimized stepper and shared
neural helper; no decoder is replaced by a cached prediction.

This is an opt-in manual experiment, separate from serving. The owner accepts
one fixed batch/context/horizon of finite, equal-length, contiguous CUDA FP32 histories.
Calls are serialized; concurrent use of one owner is unsupported. Dynamic shape
caches, ragged masks, per-block joins, admission and request cancellation are not
implemented. Construct another owner when geometry changes.

## Capture and ownership

Dense prefill uses a known-position hook to avoid upstream CUDA-scalar slice
bounds. Paged execution freezes every prefill/continuation metadata tensor and
holds independent `(series, sign)` leases until owner destruction. All warmup,
capture, replay and teardown use the arena's owning CUDA stream.

GPU initialization and counter/statistics resets are captured on every forecast.
Paged prefixes are overwritten before use; Python trace state is not replayed.
The graph is synchronized and destroyed before its plans or leases are freed.
Leases deliberately remain held between forecasts, including the eager control.

Captured kernels keep the original parameter addresses and precision settings.
Replacing a model parameter or changing TF32 or matmul precision settings after
capture can silently change eager results while graph replays continue using the
captured configuration. Close and recreate the owner after either change.

Both modes use the same `run` wrapper: validate finite inputs, copy into the
static CUDA input, execute the complete eager body or replay its graph, and
clone an independently owned output. Validation, staging and cloning remain
outside the captured graph and inside the measured call. Returned outputs are
not aliases of the graph's reusable output.

## Manual validation

Use the pinned Linux TimesFM environment, cached checkpoint and an idle GPU.

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_DIR="$PWD/results/full-graph-triton-ieee"
export VTSFM_RUN_TIMESFM_FULL_GRAPH=1 VTSFM_RUN_TIMESFM_NATIVE_KV=1
export PYTHONPATH=src:tests:.
.venv-timesfm25/bin/python -m pytest tests/test_timesfm_full_graph.py -q \
  --basetemp results/full-graph-validation-NEW
```

The checks cover A→B→A changed inputs, retained output ownership, independent
rows, batches of 1/2/8, C31/32/33/512/8192 and H1/128/129/257/1024. All nine
quantiles use `rtol=atol=1e-4`. They also reject changes to input shape, device,
dtype, contiguity and finiteness, and check paged resource release after close
or a capture failure. Full-model tests run manually on an idle GPU.

## Matched performance comparison

```bash
.venv/bin/python -m benchmarks.timesfm_full_graph --help
.venv-timesfm25/bin/python -m benchmarks.timesfm_full_graph \
  --backend all --contexts 512 --horizons 1024 --rows 2 \
  --requests 100 --warmups 5 --setup-warmups 3 --repetitions 4 \
  --data-dir data --output results/full-graph-comparison-NEW
```

Four arms (`dense-eager`, `paged-eager`, `dense-graph`, `paged-graph`) rotate
through all order positions, using fresh processes. They share the graph-safe
prefill and fixed owner/metadata contract. The controls differ from the earlier
lifecycle-inclusive [native KV study](timesfm-native-kv.md); use the new controls
to attribute improvements to graph execution.

The harness checks A/B/A against independent native forecasts before timing.
It verifies real `CUDAGraph` instances and replay counts, records raw samples and
source/data/environment hashes, and checks GPU ownership before/after calls
outside timing. Failed/contended points retain partial samples but no successful
summary. Owned processes are cleaned up on failures/cancellation.

Setup/warmup/capture time and Torch setup peaks are reported separately. Setup
NVML peaks are not measured. Steady-state Torch and sampled NVML peaks include
held graph/private pools, metadata and paged storage. Input preparation, native
reference checks and destruction are excluded. Results are serialized in-process
forecasts, not HTTP or saturated serving throughput.

For a separate activity trace, use a new output directory with `--requests 2
--warmups 1 --setup-warmups 1 --repetitions 1 --profile`. Traces run after timing
and must show actual CUDA graph launches in graph arms. Two samples are a smoke
check, not performance evidence; graph replay need not reduce kernel counts.
