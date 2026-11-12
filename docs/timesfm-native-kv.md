# TimesFM native-attention/KV feasibility (#14)

This opt-in experiment executes TimesFM through vLLM's Triton attention backend,
official physical KV allocator and `KVCacheManager`. It does **not** implement
native `GPUModelRunner`/EngineCore scheduling or a new HTTP mode. Normal serving
and forecast-block serving are unchanged.

The [measured investigation](baselines/2026-11-12-native-kv/README.md) records the
decision, source revisions, comparisons and limits. Reproduce at the recorded
revision: this prototype uses the original adapter stepper's private hooks.
Newer block executors need separate integration, not removal of the guard.

## Execution and ownership

`native_stepping.py` changes attention only on a dedicated runtime instance and
restores its forward methods on close. It retains the original tokenizer,
transformers/FFNs, weights, heads, outer/cumulative normalization, four-patch
feedback, sign statistics and forecast postprocessing. Attention preserves
**RoPE → Q/K RMS → learned Q scaling**, with backend scale **1.0**.

Whole leading masked **patch tokens** are omitted after original normalization;
raw histories are never trimmed first. This preserves the normalization width
and partially observed patches while mapping valid positions to ordinary causal
attention. Prefill supplies valid context patches; each later 128-point forecast
block supplies four queries. Packed metadata supports different series positions.

`paged_cache.py` creates a fixed physical pool with 16-token pages. Each series
has a vLLM lease covering separate normal/flipped layer views: paired signs share
an atomic lifetime and block indices, never KV storage. Twenty layers and both
signs form forty views, each `[pages,16 heads,16 patch tokens,160 K/V values]`.
One metadata plan serves the current cohort and is invalidated on lease changes.
Completion/cancellation returns leases; close drops the physical pool. All work
and page reuse use one CUDA stream.

Upstream-compatible `DecodeCache` objects hold counters and empty dense K/V.
Their `state.cache_bytes` is therefore **not** the native footprint. Report
`arena.cache_bytes` (fixed pool, including unused/null pages), leased-page bytes,
and actual allocated/reserved/device peaks separately.

## Manual checks

Use the Linux TimesFM environment and an idle GPU; dependency/checkpoint pins
are unchanged. `TRITON_F32_DEFAULT=ieee` must precede Triton import/compilation.
Torch's TF32 flags alone do not specify Triton `tl.dot` input precision.

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_DIR="$PWD/results/native-kv-triton-ieee"
export VTSFM_RUN_TIMESFM_NATIVE_KV=1 PYTHONPATH=src:tests:.
.venv-timesfm25/bin/python -m pytest tests/test_timesfm_native_kv.py -q -s \
  --basetemp results/native-kv-checks-NEW
```

Checks cover FP32 causal append, separate signs/requests, release/reuse,
patch/block boundaries, ragged histories, C8192/H1024 and mixed decode positions.
All nine forecast quantiles use unchanged `rtol=atol=1e-4`. These tests remain
manual; the small Modal contract job does not qualify this prototype.

## Matched comparison

Help is CPU-safe. GPU runs require cached weights and, with `--data-dir`, a
previously prepared AssetOpsBench fixture; the harness downloads nothing.

```bash
.venv/bin/python -m benchmarks.timesfm_native_kv --help
.venv-timesfm25/bin/python -m benchmarks.timesfm_native_kv \
  --backend all --contexts 512 --horizons 1024 --rows 2 \
  --requests 100 --warmups 5 --repetitions 3 --data-dir data \
  --output results/native-kv-comparison-NEW
```

Each backend/repetition uses a fresh process. Timed calls include initialization,
all forecast blocks, finalization/release and CUDA synchronization. Startup,
input preparation, parity checks, pool creation, GC and pool destruction are
excluded. Results are **sequential in-process forecast latency**, not HTTP or
concurrent scheduling throughput. The fixed pool remains in measured memory;
20 ms NVML samples can miss transients. Raw samples, source/package hashes,
input checksums and failure/cancellation records are retained. Outputs are
parity-checked before timing; not every timed forecast is archived/rescored.

`--profile` records one additional warmed forecast after timing. Its Chrome
trace and kernel/copy counts are separate diagnostics. Traced durations do not
replace untraced latency, and profiling allocations do not overwrite timed peaks.

`--control-checkout /path/to/optimized-adapter` adds a fourth, rotated stepper
control. Copy this exact benchmark file there first; its hash is checked. The
same Python environment runs each checkout's code. Four repetitions balance all
four order positions. The measured control is PR #61 at its recorded immutable
commit; the copied harness is its only intended working-tree difference.

## Remaining engine work

Current pooling markers describe whole forecasts/forecast blocks, rather than
patch tokens and isolated series/sign streams. They cannot be reused as native
KV attention metadata. A production port needs explicit patch-level scheduling,
model/runner KV registration, series/sign isolation, numerical-state ownership,
completion/assembly, cancellation and admission accounting. Integration with the
optimized block executor and matched HTTP/load/capacity validation are separate
work. Backend reuse alone establishes neither engine enablement nor a new kernel.
