# Native Moirai multivariate attention experiment

Issue #47 has an explicit native-only kernel experiment in
`src/vtsfm/adapters/moirai2/multivariate_attention.py`. It does not add a public
multivariate request schema, change grouping/cross-learning semantics, or enable
a serving optimization. Moirai's public adapter still rejects grouped rows and
forecasts independent univariate series. The public multivariate contract is a
separate decision.

The explicit installer patches attention forwards on one evaluation-mode native
`Moirai2Module` instance and returns an undo function. It checks the pinned source
bodies before modifying any instance, rejects an existing instance patch, and
leaves model weights, classes, scaling, projections, RoPE, and forecast code
unchanged. Imports remain CPU-safe until installation/runtime execution.

```python
import torch
from vtsfm.adapters.moirai2.multivariate_attention import install_multivariate_attention

# native_module is an official pinned Moirai2Module with FP32 weights.
native_module.eval()
undo = install_multivariate_attention(native_module)
try:
    with torch.inference_mode():
        # Use the official native module's processed tensor arguments, or a
        # supported official native forecasting wrapper. No new request format.
        output = native_module(target, observed_mask, sample_id, time_id,
                               variate_id, prediction_mask, training_mode=False)
finally:
    undo()
```

The module retains its upstream boolean mask:
`sample_id[q] == sample_id[k]` and `time_id[q] >= time_id[k]`. This includes
sample ID zero; zero is not newly interpreted as an invalid attention token.
The installer constructs one head-shared FlexAttention `BlockMask` from that
exact boolean tensor per native module call and reuses it across layers. It
resets this cache before every call and clears it after each completed or failed
call, so an installed idle model does not retain request mask buffers.
Equal/repeated timestamps across variates therefore retain their native
visibility instead of becoming token-index causality.

The score modifier runs after query/key score scaling. For each head it adds
the literal native binary bias: embedding row 0 for different variates and row
1 for equal variates. It keeps the bias rather than treating it as a constant
shift. Leading native batch axes, including the nine-trajectory axis used by
recursive forecasts, are flattened to FlexAttention's four-dimensional input
and restored afterward.

CUDA calls use `torch.compile(flex_attention, fullgraph=True, dynamic=False)`
with IEEE FP32 dot products. The pinned Torch 2.14 option is the quoted source
literal `{'FLOAT32_PRECISION': "'ieee'"}`, matching its own
`get_float32_precision()` code-generation convention. Per-layer bias weights, IDs, and mask tensors
are captured tensor inputs rather than frozen values; a CPU Dynamo check sees
one shared Flex graph across three layers and changed calls of the same shape.
That CPU check does not establish CUDA compilation or numerical parity.
Nonzero attention dropout, training, gradient-enabled calls, unsupported bias
configurations, and non-FP32 inputs are rejected. The experiment has no backward
or mixed-precision contract.

This replaces the per-layer dense floating variate bias and its masked version.
It does **not** remove all quadratic allocations: the upstream shared boolean
mask and packed scaler remain quadratic, and this pinned `create_block_mask`
implementation constructs transient head-shared boolean mask data. CPU
FlexAttention uses an unfused score-matrix implementation solely for semantic
checks. No latency, memory, HTTP, or serving benefit is established here.

## Native recursive-wrapper prerequisite

The pinned `Moirai2Forecast.forward` has a pre-existing multivariate recursive
shape failure. With the exact small-checkpoint configuration (384 model width,
6 layers, patch size 16, 4 prediction tokens), valid native input `[1, 33, 2]`
and horizon 65 fail at `uni2ts/model/moirai2/forecast.py:364`. The first recursive
update receives `[1, 8, 9, 16]` because there are two variates, while its
`rearrange` fixes `predict_token=4`; it raises `Shape mismatch, 8 != 4`.
This reproduces with a pristine randomly initialized native model, no installer
imported, and the exact pinned checkpoint configuration. We do not repair the
independent native control or claim multivariate recursive forecast parity.

First-block multivariate forecasts compare directly with the pristine official
wrapper. Official univariate recursive forecasts exercise the genuine native
nine-trajectory/81-sample quantile recursion. A separate small multivariate
module rollout checks recurrent tensor inputs against an independent pristine
module, including quantile feedback, but is a kernel contract check rather than
an implementation or validation of a multivariate forecast API. One-ULP native
attention-output perturbations provide a separate floating-noise diagnostic;
they do not relax the candidate's `rtol=atol=1e-4` gate.

## Focused checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests .venv/bin/python -m pytest \
  tests/test_moirai_multivariate_attention.py tests/test_forecast_io.py \
  tests/test_moirai_runtime.py -m 'not gpu' -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests .venv-moirai2/bin/python -m pytest \
  tests/test_moirai_multivariate_attention.py -q
VTSFM_TEST_MOIRAI_MULTIVARIATE_ATTENTION=1 \
  VTSFM_MOIRAI_CHECKPOINT=/path/to/snapshots/30f43ff08c8494f4943ae1521e9d4e94a0fbb389 \
  VTSFM_ATTENTION_RESULTS=results/issue47/gpu-correctness \
  TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/moirai-multivariate-inductor" \
  TRITON_CACHE_DIR="$PWD/.cache/moirai-multivariate-triton" \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests .venv-moirai2/bin/python -m pytest \
  -m gpu tests/test_moirai_multivariate_attention_gpu.py -q -s
```

The official tiny-model CPU tests cover packed independent samples, sample zero,
nonmonotonic timestamps, head-specific same/cross biases, multiple variables,
left padding/missing observations, univariate recursive horizons, a changed
input A→B→A sequence, unchanged caller inputs, installer restoration, and source
guards. Dropping the variate bias fails the unchanged parity tolerance by a
nontrivial margin. The original native dense bias builder is forbidden in the
candidate attention test.

The GPU suite requires the explicit environment opt-in and a locally cached
pinned checkpoint; it does not download weights. It defaults to an 18% Torch
memory cap and refuses a higher fraction. Each independent test resets its own
Dynamo cache before execution; this is not a compiler state-persistence test.
It saves raw inputs/outputs, case failures, source/checkpoint hashes, dirty state,
FP32 flags, compiler activity, Torch allocator peaks, and GPU process snapshots
to the selected ignored results directory.

## Recorded CUDA correctness

On 2026-11-11, all **3 GPU tests passed** on an A100-SXM4-40GB with Torch
2.14.0+cu130 and the pinned uni2ts/checkpoint revisions above. An existing
unrelated process was reported at 7480 MiB in the before/after process snapshots
of the shared, untimed correctness run and was left running. The test process exited after the run; the final
process snapshot contained only that original process. The IEEE option follows
the pinned quoted-literal convention described above. An earlier attempt with
an unquoted literal failed in Triton code generation; its failure record is
retained separately rather than treated as a successful run.

| Numerical check | Maximum absolute difference from pristine FP32 native |
| --- | ---: |
| Tiny packed multivariate attention, including sample zero/repeated times | 2.39e-7 |
| Checkpoint 3-variable forecast, C=65/H=64, ordinary and ragged | 9.54e-7 |
| Checkpoint univariate recursive forecast, C=65/H=129, ragged | 3.94e-6 |
| Checkpoint changed-input comparison and return to the earlier shape | 1.44e-6 |
| Tiny multivariate native-module tensor feedback, three blocks | 4.77e-7 |

All positive comparisons pass the unchanged `rtol=atol=1e-4` gate. The separate
one-ULP native-output noise baseline passes the same gate across three feedback
blocks. Dropping the variate bias fails that gate: the maximum absolute change
is 1.054 on the tiny packed case and 0.117 on the learned checkpoint forecast.
The checkpoint test also retains the pristine multivariate recursive-wrapper
failure described above; it is not counted as supported forecast behavior.

A single untimed activity trace records `triton_tem_fused_flex_attention_0` and
its fused follow-up kernels, establishing actual CUDA kernel execution. Compiler
counters show 4/4/2 unique graphs in the three independent tests, with no recorded
graph breaks or unimplemented operations. The CPU structural test separately
checks graph sharing across layers. These observations do not establish a
serving speedup, memory reduction, or general compiler shape-state guarantees.

Raw JSON/NPZ evidence is retained locally in
`results/issue47/gpu-correctness-final`; the initial code-generation failure is
under `results/issue47/gpu-correctness-1`. Generated artifacts and weights remain
outside Git. Memory/performance measurements still require an idle GPU and
matched controls. Public multivariate semantics and the upstream
recursive-wrapper prerequisite remain outside this kernel patch.
