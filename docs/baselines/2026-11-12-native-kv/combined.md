# Optimized TimesFM dense/paged ablation — November 12, 2026

## Question and scope

**Status: implementation validated; clean performance comparison pending an
uninterrupted GPU window.** Another workload arrived during the first attempt.
That attempt is retained as failed provenance and excluded from the comparison.

Does paged attention/cache handling help when both implementations use PR #61's
optimized execution? The [original study](README.md) compared different flip
orchestration. This follow-up removes that difference and adds a narrower
precomputed-QKV diagnostic. It remains a manual, in-process backend experiment,
not native GPUModelRunner/EngineCore or HTTP enablement.

## Matched implementation

Validated implementation and interrupted attempt:
`04f8eb60eabffb62c9a1a6e4b420f1a3124d09e1`, clean checkout.
The guarded rerun harness is `62f61f1f09aa460bd4f0bdbbfe73779e4aa099c1`
(model/executor source is unchanged).
The branch reuses PR #61's TimesFM executor/tests at
`58a3c843984c932f0eaf837a79fa496872260120`, adding an optional attention boundary.
PR #61 must land first; its copied source overlap should be resolved before
merging this experiment. No other changes from that PR are assumed by this study.

Both forecast paths share `_advance_passes` and `block_decode_forward`, including
normal/flipped fusion, running statistics, tokenizer, QKV, RoPE, Q/K RMS, learned
query scaling, projections, FFNs, both heads and forecast assembly. Both keep two
separate sign calls during prefill. The optional boundary supplies metadata,
cache append and attention. The paged path has one disjoint lease per
`(series, sign)` in one physical layer view, allowing one attention call/layer for
all signs. Tests verify that call geometry, ownership and counter behavior.

The full-forecast comparison includes each backend's initialization, cache
allocation/leases, metadata, prefill, append/gather and cleanup costs. It isolates
the integrated attention/cache implementation, rather than an intrinsic kernel.
The paged pool is created outside timing but remains included in memory peaks.

## Environment and workload

The A100-SXM4-40GB, driver 595.91.07 and pinned TimesFM runtime/checkpoint match
the [original provenance table](README.md#sources-and-method): vLLM 0.33.0,
Torch 2.14.0+cu130/CUDA 13.0, TimesFM package 3.0.2 at `e51928e`, eager FP32,
TF32 disabled, `TRITON_F32_DEFAULT=ieee`, four Torch threads. The VM has 12
logical CPUs and remains shared. The previously CPU-only process became
GPU-active during the first attempt. Measurement requires exclusive GPU access;
the interrupted attempt lost exclusivity. No drivers/services were changed.

Full forecasts use the same cached observed/causally imputed AssetOpsBench
MetroPT fixture, B2/C512/H1024, eight forecast blocks and all nine quantiles.
Both rows' input hash is
`db459fe68b795882f093e47e2679155b220e12e528e543e7ce4c24f87b65f596`.
The planned four rounds alternate dense/paged order. Each path/round uses a fresh
process, five warmups and 100 timed forecasts: a target of 400 samples/path,
800 total. These totals have not yet been completed without contention.

Timing includes initialize, all blocks, finalize/release and CUDA synchronization.
Startup, input preparation, parity, pool creation/destruction and profiling are
excluded. These are serialized in-process B2 forecast latencies and reciprocal
throughput, not HTTP or saturated-server measurements. P95 pools raw samples
using linear interpolation; it does not average per-round p95s. Torch peaks reset
after warmup/GC/empty-cache; NVML samples every 20 ms and may miss transients.

## Full-forecast results

The first attempt completed seven children; the eighth failed its initial idle
check before model loading. The fourth-round paged point's sampled device memory
had already risen from the earlier roughly 2036 MiB to 2589–2931 MiB, confirming
overlap with another workload. The entire incomplete attempt is excluded from
primary results. No final mean/p95 or speedup is reported from it.

The updated harness checks NVML compute/graphics PIDs before/after each timed
sample outside timing. Newly arriving foreign work fails the point and removes
successful summaries while retaining raw samples and ownership observations.
These snapshots do not reserve the GPU or exclude brief activity between checks.
A fresh four-round set is required once the node is free.

## Precomputed attention diagnostic

This separate checkpoint-free synthetic experiment uses identical precomputed
FP32 Q/K/V with 16 heads, head size 80, scale 1 and four causal queries. Prefixes
16/64/256 patches correspond to 512/2048/8192 observation positions. Internal
rows 2/4/16 correspond to 1/2/8 independent series when both signs are included.
Native TimesFM SDPA and vLLM's conservative 2D Triton backend see the same valid
prefix, with no extra unused dense cache capacity.

Each of the nine shapes has four rotated timing paths: native/paged attention
alone and native/paged four-slot update plus attention. Each path has four rounds,
100 samples/round and 50 operations/sample, with ten warmups. The same four slots
are rewritten on every replay; this does not exercise growing-cache lifecycle.
Metadata, leases, tensor preparation, normalization/QKV/FFNs/heads and scheduling
are excluded. Both backends coexist; no independent memory comparison is made.

Eager timing includes output allocation on both paths. A separate CUDA-graph run
captures each fixed operation outside timing, retains static outputs and checks
replay parity. Graph timing reduces Python launch work; it is not full-forecast
graph support. CUDA events measure stream elapsed, including possible launch
gaps, rather than summed kernel duration. Wall time includes event synchronization,
amortized over 50 operations. Its sample p95 is not a request tail percentile.

Eager and graph-replay smoke checks passed output parity at P16/R4. They used only
two samples per path and establish harness correctness, not a speedup or tail
latency result. The full grids were not started because the preceding forecast
run failed; they remain pending GPU availability.

## Separate trace

One extra warmed full forecast per path was traced after a two-sample harness
smoke, separate from all measurements above:

| Path | CUDA kernel events | GPU copy events | GPU copy bytes |
| --- | ---: | ---: | ---: |
| Optimized dense | 21,927 | 542 | 25,392 |
| Optimized paged | 19,486 | 381 | 26,534 |

GPU memcpy categories omit strided copies implemented as kernels; their byte
counts do not quantify all dense KV gather traffic. Fewer events do not imply
lower latency. Traced duration is not substituted for untraced latency.

## Validation and interpretation

The clean measured source passed 66 native/backend/tiny-model checks, including
full C8192/H1024 parity, ragged histories, mixed positions, reordering, early
completion, cancellation/reuse and one fused attention call per continuation
layer. The shared optimized executor also passed 22 real-model stepping checks
and three HTTP lifecycle/capacity checks before measurement. Local CPU checks
passed 534 tests with 18 optional-runtime skips, including the ownership guard.
Ruff, package build and fresh installed-wheel CPU-safe discovery passed.

Initial validation omitted a copied PR #61 fixture, causing an import failure;
its retained CUDA process then made same-process HTTP idle checks refuse setup.
The missing fixture was restored and stepping/HTTP suites passed in separate
processes. A diagnostic attempted while those checks were active was rejected by
the idle guard before doing GPU work. These setup records are retained alongside
the successful runs; they are not counted as benchmark samples.

The shared executor and fused attention call geometry are now established.
Whether paged execution improves full forecasts or the narrower backend replay
is still an open measurement question. The original separate-sign result is
insufficient to answer it; neither that result nor the interrupted attempt should
be presented as the matched ablation's conclusion.

These results qualify one workload, these implementations and this GPU. They do
not qualify tuned 3D/split-K attention, different precisions, fragmented capacity,
HTTP SLOs, other models or a new kernel. Keep full-forecast speed, backend replay
speed and cache capacity claims separate.

## Reproduce and artifacts

Use the [manual guide](../../timesfm-native-kv.md) at the measured source, with
the unchanged pinned Linux TimesFM environment and cached data/weights:

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 TRITON_F32_DEFAULT=ieee
export VTSFM_RUN_TIMESFM_NATIVE_KV=1 PYTHONPATH=src:tests:.
export TRITON_CACHE_DIR="$PWD/results/combined-triton-ieee"
.venv-timesfm25/bin/python -m benchmarks.timesfm_native_kv \
  --backend ablation --contexts 512 --horizons 1024 --rows 2 \
  --requests 100 --warmups 5 --repetitions 4 --data-dir data \
  --output results/combined-forecasts-NEW
.venv-timesfm25/bin/python -m benchmarks.timesfm_attention_ablation \
  --prefixes 16,64,256 --rows 2,4,16 --samples 100 --iterations 50 \
  --warmups 10 --repetitions 4 --output results/attention-eager-NEW
.venv-timesfm25/bin/python -m benchmarks.timesfm_attention_ablation \
  --prefixes 16,64,256 --rows 2,4,16 --samples 100 --iterations 50 \
  --warmups 10 --repetitions 4 --cuda-graphs --output results/attention-graphs-NEW
```

The interrupted attempt, source bundle, successful validation/HTTP artifacts,
separate traces and setup-failure records are retained remotely under
`~/vtsfm-native14-combined-20261112/`. The ignored local copy is
`results/native-kv-combined-20261112/partial-evidence.tar.gz`, SHA256
`4edfa78b9d2e1160de44c2ea3eff157c994efc18fb5190b3aab526dd2eb8dbd2`.
The later ownership probe used NVML only, imported neither Torch nor vLLM and
correctly rejected the foreign process without GPU computation. Raw runs,
traces and source bundles remain outside Git.
