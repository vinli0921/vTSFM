# TimesFM vLLM attention/KV feasibility — November 12, 2026

## Decision

**Technical feasibility passes; defer the production runner port.** Real vLLM
block leases, physical cache allocation and Triton attention preserve the
recorded TimesFM forecasts. On the measured workload, however, the prototype
takes 756.5 ms per forecast versus 380.0 ms for the optimized adapter control.
Prioritize that adapter's block execution and the separate compile/graphs work.
Retain this opt-in proof for a smaller follow-up before committing to a full port.

This establishes a backend/allocator boundary, not native GPUModelRunner/
EngineCore or HTTP serving support. It does not show that an optimized native
port cannot win: the implementations differ in flip orchestration and the native
backend is an untuned eager FP32 proof.

## Sources and method

| Item | Revision/settings |
| --- | --- |
| Baseline source | `c2b9f4580f61ee18e7047620dc3667f371803582` |
| Measured prototype/harness | `be7fd686f9670e3978b3ce0e128ae65ed1de4aea` |
| Optimized adapter control, PR #61 | `58a3c843984c932f0eaf837a79fa496872260120` |
| TimesFM upstream | `e51928e27119cb17bebc005be2696b75e0a9e688` (package 3.0.2) |
| TimesFM checkpoint | `google/timesfm-2.5-200m-pytorch`, `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| GPU | A100-SXM4-40GB, driver 595.91.07 |
| Runtime | Python 3.11.17, vLLM 0.33.0, Torch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0 |
| Precision | Eager FP32, TF32 disabled, `TRITON_F32_DEFAULT=ieee`, four Torch threads |
| Workload | B2/C512/H1024, eight 128-point forecast blocks, all nine quantiles |

The same cached AssetOpsBench MetroPT fixture supplies both independent rows.
It contains observed data and documented causal forward-fill; windows/sensors
are correlated. This is a repeated fixed-input performance/parity experiment,
not observed-label forecast-quality evaluation.

Four rounds rotate the four implementations through every order position.
Each path/round gets a fresh process, five warmups and 100 timed forecasts.
This yields 400 samples per path and 1,600 total. Input checksums, checkpoint,
environment and timing boundaries match. For this in-process TimesFM comparison,
the exercised model path differs in stepping/block-decoding code; the reference
tensor runtime, model settings and weights are identical. The full PR #61 source
also differs in unrelated serving/Moirai files. Its only intentional dirty file
is the identical copied benchmark harness; the primary source is clean.

Timing covers one sequential in-process forecast, including state initialization,
all blocks, finalization/release and CUDA synchronization. Startup, input
preparation, parity checking, pool creation/destruction, GC and profiling are
excluded. This is not HTTP p95, offered-load throughput or four concurrent HTTP
requests; those boundaries differ from earlier serving reports.

Torch peaks are reset after warmup/GC/empty-cache outside timing. Native fixed
pool bytes remain included. NVML samples the device every 20 ms and can miss
transients. The GPU is exclusive to the owned benchmark; the 12-logical-CPU VM
is shared, with an existing CPU-only Python process observed and left untouched.

## Repeated untraced results

All 16 fresh-process points completed successfully: 100 samples per point,
400 per path and 1,600 total. P95 pools the 400 raw samples using linear
interpolation; it is not the average of the four per-round percentiles.

| Path | Mean ms | Pooled p95 ms | Four-round mean range ms | Sequential forecasts/s |
| --- | ---: | ---: | ---: | ---: |
| Whole tensor | 692.845 | 709.692 | 685.646–698.023 | 1.443 |
| Original stepper | 741.522 | 753.173 | 729.441–747.291 | 1.349 |
| Paged prototype | 756.498 | 783.193 | 742.153–769.942 | 1.322 |
| Optimized adapter, PR #61 | 380.003 | 390.473 | 370.265–384.801 | 2.632 |

These are sequential **B2 forecasts**, each returning two series, rather than
saturated server throughput. The paged implementation's mean is 9.19% above the
tensor baseline, 2.02% above the original stepper and 99.08% above the optimized
control. Its copy reduction does not produce a latency win in this experiment.

Maximum observed peaks across the four rounds, in MiB:

| Path | Torch allocated | Torch reserved | Sampled device/NVML |
| --- | ---: | ---: | ---: |
| Whole tensor | 913.236 | 982 | 2012.563 |
| Original stepper | 959.851 | 1042 | 2072.563 |
| Paged prototype | 938.892 | 1008 | 2038.563 |
| Optimized adapter, PR #61 | 929.263 | 1004 | 2034.563 |

The paged pool is 43.75 MiB, including unused/null pages, and remains charged in
these peaks. All paged leases were zero after trials and cleanup in every round.
All pre-timing parity checks passed: maximum absolute error was zero for tensor/
original and `9.5367431640625e-7` for paged/optimized, over 18,432 values per
point. Every recorded source hash matches its checkout; inputs, installed-source
hashes, checkpoint, precision and GPU match across all points. Cleanup stopped
all 16 owned children without errors; the GPU was idle afterward.

The decision is limited to this fixed B2/C512/H1024 workload and these complete
implementations on one shared-CPU VM. The native proof uses conservative 2D Triton
attention without split-K tuning. No attention-kernel-only ablation, capacity
advantage, long-context performance sweep or offered-load SLO result is claimed.

## Separate traced diagnostic

One additional warmed forecast per implementation was traced outside timing.
These are operation counts, not the untraced latency measurements above.
The diagnostic used the same implementation source hashes before the source
commit was recorded; its working tree is explicitly marked dirty.

| Path | CUDA kernel events | GPU copy events | GPU copy bytes |
| --- | ---: | ---: | ---: |
| Whole tensor | 39,452 | 1,603 | 16,896 |
| Original stepper | 39,623 | 4,111 | 540,728,592 |
| Paged prototype | 34,753 | 1,346 | 49,986 |
| Optimized adapter, PR #61 | 21,927 | 542 | 25,392 |

Copy counts include GPU memcpy events in all directions, including small
metadata transfers. The original stepper's dense KV merge/store copies dominate
its byte count. Paged KV removes that traffic, but substantial eager execution
remains. The optimized adapter also fuses normal/flipped continuation calls;
the prototype retains the original separate calls. Therefore the optimized
control is a practical alternative, not an isolated attention-kernel ablation.

## Correctness and lifetime

Clean prototype source passed eleven manual checks: nine CUDA cases and two
CPU-safe CLI checks. Coverage includes FP32 causal append, distinct requests/
signs, page release/reuse, full forecast parity at C8192/H1024, ragged histories,
mixed physical decode positions, reordering and early completion. All nine
quantiles use unchanged `rtol=atol=1e-4`.

A timed benchmark cancellation retained 61 completed samples, preserved its
partial point, cleared KV leases and stopped all owned children. An earlier
probe interrupted startup, correctly retaining a zero-sample cancellation
record; its checker expected a timed cancellation too early. Both records are
retained. Local CPU checks passed 527 tests (15 optional-runtime skips); lint,
source/wheel build and installed-package checks are recorded with the PR.

## Scope of a production follow-up

The current serving wrapper exposes no vLLM attention layers; its upstream model
owns attention internally. Its one-per-forecast/block prompt markers do not
represent native patch-token positions or isolated series/sign streams. Using
their metadata for native attention would be incorrect.

A separately scoped port would need:

1. Native model/KV registration and a patch-level execution contract, including
   isolated series/sign sequences, multi-query continuation and final forecasts.
2. Runner/scheduler/IO mapping, cancellation, output assembly and admission/memory
   accounting for those internal sequences.
3. Integration with the optimized block executor, then native/HTTP parity,
   staggered lifecycle, offered-load and capacity qualification.

Estimate: one to two developers for roughly two to three weeks after a narrow
design/source review, with uncertainty around pinned V1 pooling/KV interactions.
A smaller next experiment could combine flip orchestration in the paged proof
and tune backend metadata/scratch before deciding on that port.

Other adapters need independent analysis: Toto mixes temporal/variate attention
and replaces forecast feedback positions; Moirai's recursive trajectories and
normalization may invalidate simple append-only KV reuse. This experiment
qualifies neither family and introduces no new kernel or conference novelty.

## Reproduce

Use the [manual guide](../../timesfm-native-kv.md), the recorded prototype source,
and the pinned TimesFM environment. Prepare a separate control checkout at the
PR #61 revision and copy the exact harness into it, without modifying its
runtime. With weights and AssetOpsBench data already cached:

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4 TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_DIR="$PWD/results/native-kv-triton-ieee"
export VTSFM_RUN_TIMESFM_NATIVE_KV=1 PYTHONPATH=src:tests:.
.venv-timesfm25/bin/python -m pytest tests/test_timesfm_native_kv.py -q -s
.venv-timesfm25/bin/python -m benchmarks.timesfm_native_kv \
  --backend all --contexts 512 --horizons 1024 --rows 2 \
  --requests 100 --warmups 5 --repetitions 4 --data-dir data \
  --control-checkout /path/to/pr61-control \
  --output results/native-kv-counterbalanced-NEW
```

Run a separate diagnostic with `--requests 1 --warmups 1 --repetitions 1 --profile`
and a new output directory. Its one-sample p95 is not a tail-latency result.
Raw artifacts, Chrome traces, hashes, manifests and cancellation/cleanup evidence
are retained under the manual run's `~/vtsfm-native14-20261112/` directory and in
the ignored local `results/native-kv-20261112/`. The transferred `evidence.tar.gz`
has SHA256 `8454608a9917b33db8cebbe3ba834acec0c312db345578220af90f10a07017a8`;
it includes source bundles for both measured revisions, not model weights or
environments. The local derived `summary.json` pools raw `samples_s` with
`benchmarks.common.summarize`; its peak columns take the maximum of the four
point records. Raw results and traces are not checked into Git.
