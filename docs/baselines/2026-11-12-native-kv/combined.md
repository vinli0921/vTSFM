# Optimized TimesFM dense/paged ablation — November 12, 2026

## Question and scope

**Decision: retain optimized dense execution for the current eager workload.**
The complete guarded comparison measures 387.5 ms dense versus 427.7 ms paged:
paged mean latency is 10.38% higher. The narrower diagnostic finds a conditional
benefit. With fixed-operation graph replay, paged update-plus-attention is 1.89×
faster than a compact dense SDPA control at a short prefix. The compact dense
control is faster at long prefixes. These results support profiling/capture and
shape-specific backend work before a production runner port.

All 800 forecasts and both nine-shape diagnostic grids completed successfully.
An earlier interrupted attempt is retained but excluded from primary results.

Does paged attention/cache handling help when both implementations use PR #61's
optimized execution? The [original study](README.md) compared different flip
orchestration. This follow-up removes that difference and adds a narrower
precomputed-QKV diagnostic. It remains a manual, in-process backend experiment,
not native GPUModelRunner/EngineCore or HTTP enablement.

## Matched implementation

Measured source: `62f61f1f09aa460bd4f0bdbbfe73779e4aa099c1`, clean checkout.
Model/executor source is unchanged from the validated implementation and
interrupted attempt at `04f8eb60eabffb62c9a1a6e4b420f1a3124d09e1`.
The measured implementation used PR #61's TimesFM executor/tests at
`58a3c843984c932f0eaf837a79fa496872260120`, adding an optional attention boundary.
The current implementation also includes PR #68's batched sign prefill. The
numbers below are the historical control with separate sign prefills; rerun
both arms before attributing any result to the current implementation.

Both measured forecast paths share `_advance_passes` and
`block_decode_forward`, including
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
Four completed rounds alternate dense/paged order. Each path/round uses a fresh
process, five warmups and 100 timed forecasts: 400 samples/path, 800 total.

Timing includes initialize, all blocks, finalize/release and CUDA synchronization.
Startup, input preparation, parity, pool creation/destruction and profiling are
excluded. These are serialized in-process B2 forecast latencies and reciprocal
throughput, not HTTP or saturated-server measurements. P95 pools raw samples
using linear interpolation; it does not average per-round p95s. Torch peaks reset
after warmup/GC/empty-cache; NVML samples every 20 ms and may miss transients.

## Full-forecast results

| Path | Mean ms | Pooled p95 ms | Four-round mean range ms | Sequential B2 forecasts/s |
| --- | ---: | ---: | ---: | ---: |
| Optimized dense | 387.455 | 406.009 | 385.223–388.723 | 2.581 |
| Optimized paged | 427.656 | 441.234 | 419.912–440.401 | 2.338 |

Paged mean is 10.38% higher and p95 is 8.68% higher. Paged mean exceeds dense in
all four rounds; the ranges are not confidence intervals. Each B2 forecast
returns two series. This does not measure saturated HTTP throughput.

Maximum observed memory peaks across rounds, MiB:

| Path | Torch allocated | Torch reserved | Sampled device/NVML |
| --- | ---: | ---: | ---: |
| Optimized dense | 929.263 | 1004 | 2034.563 |
| Optimized paged | 935.517 | 1006 | 2036.563 |

The paged pool is 40.625 MiB, including unused/null pages, and remains included
in these peaks. All pre-timing nine-quantile comparisons pass unchanged
`rtol=atol=1e-4`; maximum absolute error is `9.5367431640625e-7` for both paths.
All paged leases return to zero after trials and cleanup. All eight owned
children stopped without errors.

Each child records 202 ownership snapshots: 1,616 total, with no foreign PIDs or
query errors. Source hashes, installed-source hashes, checkpoint, precision,
geometry, GPU and input data match across all eight points. Independent remote
and local raw-sample aggregation reproduce the summaries.

### Excluded first attempt

The first attempt completed seven children; the eighth failed its initial idle
check before model loading. The fourth-round paged point's sampled device memory
had already risen from the earlier roughly 2036 MiB to 2589–2931 MiB, confirming
overlap with another workload. The entire incomplete attempt is excluded from
primary results. No final mean/p95 or speedup is reported from it.

The updated harness checks NVML compute/graphics PIDs before/after each timed
sample outside timing. Newly arriving foreign work fails the point and removes
successful summaries while retaining raw samples and ownership observations.
These snapshots do not reserve the GPU or exclude brief activity between checks.
The fresh four-round set above replaces that attempt in full; no partial samples
were reused.

## Precomputed attention diagnostic

This separate checkpoint-free synthetic experiment uses identical precomputed
FP32 Q/K/V with 16 heads, head size 80, scale 1 and four causal queries. Prefixes
16/64/256 patches correspond to 512/2048/8192 observation positions. Internal
rows 2/4/16 correspond to 1/2/8 independent series when both signs are included.
Native TimesFM SDPA uses a compact `prefix + 4` cache. vLLM's conservative 2D
Triton backend sees the same valid positions in pages. The optimized dense
forecast passes the full padded width to SDPA with a causal mask. At C512/H1024,
the first decode passes 44 cache slots to optimized dense SDPA and the diagnostic
passes 20 to compact SDPA. The graph speedup and crossover below apply to the
compact SDPA control.

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

Both full grids completed all nine shapes. Every shape/path has 400 sample
averages, each averaging 50 operations: 20,000 replays/path/shape. All ordinary
and captured-output comparisons pass unchanged `rtol=atol=1e-4`; maximum absolute
error is `6.854534149169922e-7`. QKV hashes match across eager/graph modes. All
leases return to zero, both processes stopped, and 57,933 ownership snapshots
contain no foreign PIDs or query errors.

Mean stream elapsed per operation, microseconds (lower is better):

### Eager diagnostic

| Prefix patches | Internal rows | Dense attention | Paged attention | Dense update + attention | Paged update + attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 97.480 | 152.426 | 127.366 | 276.098 |
| 16 | 4 | 98.035 | 152.013 | 128.454 | 273.011 |
| 16 | 16 | 99.349 | 168.904 | 128.982 | 276.213 |
| 64 | 2 | 99.967 | 157.081 | 129.987 | 283.823 |
| 64 | 4 | 98.876 | 156.643 | 129.089 | 282.867 |
| 64 | 16 | 99.491 | 156.610 | 129.609 | 282.640 |
| 256 | 2 | 99.287 | 157.208 | 129.254 | 283.515 |
| 256 | 4 | 99.091 | 157.097 | 129.325 | 282.548 |
| 256 | 16 | 167.052 | 210.955 | 174.517 | 286.621 |

Paged attention alone is 1.26–1.70× slower in all nine eager shapes; including
updates it is 1.64–2.19× slower. These timings retain backend Python/launch gaps.

### Fixed-operation CUDA-graph diagnostic

| Prefix patches | Internal rows | Dense attention | Paged attention | Dense update + attention | Paged update + attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 25.333 | 12.578 | 27.571 | 14.742 |
| 16 | 4 | 23.999 | 12.747 | 28.115 | 14.912 |
| 16 | 16 | 49.568 | 28.273 | 54.085 | 31.101 |
| 64 | 2 | 27.321 | 28.582 | 31.062 | 30.832 |
| 64 | 4 | 28.177 | 28.892 | 32.071 | 31.008 |
| 64 | 16 | 59.950 | 71.727 | 64.447 | 74.452 |
| 256 | 2 | 55.960 | 76.840 | 59.791 | 78.943 |
| 256 | 4 | 56.298 | 77.382 | 60.575 | 79.522 |
| 256 | 16 | 154.362 | 205.631 | 159.079 | 209.350 |

With graph replay, the short-prefix (16) update-plus-attention path is 1.74–1.89×
faster with paging. Prefix 64 is mixed: approximately tied for two/four rows and
slower at sixteen rows. At prefix 256, paged update-plus-attention is 31–32%
slower. For four internal rows (B2), the short-prefix pair is 28.115 versus
14.912 µs; the long-prefix pair is 60.575 versus 79.522 µs, dense/paged.

This is a backend comparison including different kernels/layouts, not a
page-layout-only ablation. Graph capture changes launch/allocation behavior as
well as output ownership; it does not establish summed kernel time or a
full-forecast graph speedup. Do not infer that removing Python overhead makes
paging uniformly better.

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

The clean implementation at `04f8eb6` passed 66 native/backend/tiny-model checks, including
full C8192/H1024 parity, ragged histories, mixed positions, reordering, early
completion, cancellation/reuse and one fused attention call per continuation
layer. The shared optimized executor also passed 22 real-model stepping checks
and three HTTP lifecycle/capacity checks for the dense serving control before
measurement. The paged prototype still has no HTTP mode. Local CPU checks
passed 534 tests with 18 optional-runtime skips, including the ownership guard.
Ruff, package build and fresh installed-wheel CPU-safe discovery passed.
Model/backend hashes are unchanged in the measured `62f61f1` source; that revision
adds benchmark ownership checks and completes the guarded measurements above.

Initial validation omitted a copied PR #61 fixture, causing an import failure;
its retained CUDA process then made same-process HTTP idle checks refuse setup.
The missing fixture was restored and stepping/HTTP suites passed in separate
processes. A diagnostic attempted while those checks were active was rejected by
the idle guard before doing GPU work. These setup records are retained alongside
the successful runs; they are not counted as benchmark samples.

The shared executor removes the original flip-fusion confound. The current eager
paged implementation remains slower and uses slightly more memory on the tested
forecast workload. The graph diagnostic demonstrates a real short-prefix backend
opportunity, with a crossover rather than a universal advantage.

Keep the optimized dense adapter as the near-term serving path. Next, test
graph-ready block execution under #17 and profile metadata/launch costs; if
continuing the native proof, compare shape-specific/tuned attention before a
runner port. Those are proposed follow-ups, not implemented full-forecast graph
or adaptive backend support. The study does not isolate a single cause for the
end-to-end gap.

These results qualify one workload, these implementations and this GPU. They do
not qualify tuned 3D/split-K attention, different precisions, fragmented capacity,
HTTP SLOs, other models or a new kernel. Keep full-forecast speed, backend replay
speed and cache capacity claims separate.

## Reproduce and artifacts

Use the [manual guide](../../timesfm-native-kv.md) at the measured source, with
the unchanged pinned Linux TimesFM environment and cached data/weights:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
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

The successful guarded runs and source bundle are retained in
`guarded-evidence.tar.gz`, SHA256
`c70616a2b9b5eced04de94d53760578e68d8daefa65e061d2f1f93e553d3ba5d`,
both remotely and in the ignored local results directory. `guarded-completion.json`
records all manifests successful and the GPU idle afterward. The local
`aggregate_and_plot.py` cross-checks raw counts, parity, ownership, leases,
cleanup and matching source/settings; it produces `aggregation.json` with
39 input-JSON hashes, plus `timesfm-paged-ablation.png` and `.pdf`. The figure's
whiskers show round-mean ranges, not confidence intervals. These derived
artifacts remain ignored and are not needed to reproduce the measurements.
