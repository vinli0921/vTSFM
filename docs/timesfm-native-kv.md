# TimesFM native-attention/KV feasibility (#14)

This opt-in experiment executes TimesFM through vLLM's Triton attention backend,
official physical KV allocator and `KVCacheManager`. It does **not** implement
native `GPUModelRunner`/EngineCore scheduling or a new HTTP mode. Normal serving
and forecast-block serving are unchanged.

The [original investigation](baselines/2026-11-12-native-kv/README.md) records the
first prototype and its immutable source revisions. The
[combined follow-up](baselines/2026-11-12-native-kv/combined.md) records current
validation and benchmark status. The current experiment uses
PR #61's optimized block executor, with a small attention/cache boundary shared
by dense and paged execution. Reproduce each report at its recorded revision;
the original separate-sign prototype is at `be7fd68`.

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

`paged_cache.py` creates a fixed physical pool with 16-token pages. Each
`(series, sign)` has its own vLLM lease and disjoint block IDs. Twenty layer views
each have shape `[pages,16 heads,16 patch tokens,160 K/V values]`. Keeping signs
in the same physical view allows one attention call per layer for all normal/
flipped rows. The arena's original two-view mode remains available for backend
tests. Completion/cancellation returns both signs' leases; close drops the pool.
One cohort metadata plan is invalidated on lease changes. All work/page reuse
uses one CUDA stream.

Both full-forecast comparison paths use the same optimized `_advance_passes` and
`block_decode_forward`: flip fusion, tokenizer, QKV/RoPE/QK normalization/query
scaling, output projection, FFNs, both heads, statistics and forecast assembly
are shared. The optional backend supplies metadata preparation and cache append/
attention only. Prefill still makes two separate sign calls in both paths. The
native proof uses empty dense K/V and the same counter arithmetic as the control.

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
  --backend ablation --contexts 512 --horizons 1024 --rows 2 \
  --requests 100 --warmups 5 --repetitions 4 --data-dir data \
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

Both experimental harnesses check NVML compute/graphics ownership before and
after each timed forecast or microbatch, outside timing. Foreign activity fails
the point, preserves raw partial samples and invalidates performance summaries.
Snapshots cannot exclude very short activity between checks and do not reserve
the GPU. Arrange an uninterrupted window; exclude interrupted attempts from the
primary comparison and rerun the full counterbalanced set.

`--backend ablation` alternates the optimized dense and paged implementations.
Four repetitions balance their order. This isolates the attention/cache
implementation within shared optimized execution, including its allocation,
lease, metadata, append/gather and prefill costs; it is not a kernel-only result.

`--profile` records one additional warmed forecast after timing. Its Chrome
trace and kernel/copy counts are separate diagnostics. Traced durations do not
replace untraced latency, and profiling allocations do not overwrite timed peaks.

`--control-checkout /path/to/optimized-adapter` adds a fourth, rotated stepper
control. Copy this exact benchmark file there first; its hash is checked. The
same Python environment runs each checkout's code. Four repetitions balance all
four order positions. The measured control is PR #61 at its recorded immutable
commit; the copied harness is its only intended working-tree difference.

## Precomputed attention diagnostic

For a narrower comparison, `benchmarks.timesfm_attention_ablation` uses identical
synthetic FP32 Q/K/V with 16 heads of size 80, scale 1, four causal queries and
exactly the same valid cache prefix. It compares native TimesFM SDPA and vLLM
2D Triton attention, each with and without four-slot cache updates:

```bash
.venv-timesfm25/bin/python -m benchmarks.timesfm_attention_ablation \
  --prefixes 16,64,256 --rows 2,4,16 --samples 100 --iterations 50 \
  --warmups 10 --repetitions 4 --output results/attention-eager-NEW
# Repeat with a fresh output directory and --cuda-graphs for fixed-shape replay.
```

Rows are internal attention rows, including signs: four rows correspond to B2.
Metadata, leases, QKV/normalization, neural layers and scheduling are excluded.
The same four slots are rewritten, not appended indefinitely. Both backends
coexist, so this diagnostic makes no independent peak-memory comparison.
CUDA events report stream elapsed time, including possible CPU launch gaps;
wall time also includes the final event synchronization. P95 is across samples
that each average 50 operations, not per-request tail latency.

`--cuda-graphs` captures each fixed operation outside timing, retains static
outputs and checks replay parity. It reduces per-operation Python launch work
but does not measure summed kernel duration or enable graphs for full forecasts/
serving. Eager and graph results must remain separately labeled. All native
Triton results use conservative 2D attention, not tuned split-K scratch.

## Remaining engine work

Current pooling markers describe whole forecasts/forecast blocks, rather than
patch tokens and isolated series/sign streams. They cannot be reused as native
KV attention metadata. A production port needs explicit patch-level scheduling,
model/runner KV registration, series/sign isolation, numerical-state ownership,
completion/assembly, cancellation and admission accounting. Integration with the
optimized block executor and matched HTTP/load/capacity validation are separate
work. Backend reuse alone establishes neither engine enablement nor a new kernel.
