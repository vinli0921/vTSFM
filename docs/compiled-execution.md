# Guarded neural compilation draft

The default serving mode remains eager FP32. This first part of #17 adds an
explicit experimental neural compiler selection for TimesFM, Toto and Moirai:

```bash
./scripts/serve.sh --adapter toto2 --forecast-execution compile
```

The CLI sets the compiler request, disables Triton FP contraction and disables
TF32 in the child environment before Torch imports. Direct runtime users must
set `VTSFM_FORECAST_EXECUTION=compile`, `TRITON_DEFAULT_FP_FUSION=0` and
`NVIDIA_TF32_OVERRIDE=0` before importing Torch. Discovery, CLI help and dry-run
remain CPU-safe. Existing `--enforce-eager` remains set for the vLLM runner;
compilation applies only to the selected adapter's neural callable.

| Adapter | Compiled callable | Work left outside the callable |
| --- | --- | --- |
| TimesFM | `model.forward` | Whole-forecast preparation, decode loop and quantile correction |
| Toto | `model.transformer.forward` | Scaling, patch/output projections and forecast feedback |
| Moirai | `model.encoder.forward` | Scaling/projections, recursive quantile trajectories and existing NumPy wrapper |

The mode requires the pinned Torch 2.14.0 and FP32 parameters with highest matmul
precision. Reduced precision and continuous batching are rejected explicitly.
No checkpoint, HTTP response or model-forecast settings change.

## Native boundaries and ownership

Unrestricted compilation failed the research copy's strict Moirai long-horizon
parity check. This implementation therefore keeps pinned native RMSNorm,
attention and activation/FFN boundaries eager with `torch.compiler.disable`.
Third-party forward sources are checked against the pinned source bodies before
any method is changed. Attention masks, normalization, unit scaling and native
activations are not reimplemented. Only instance methods are changed; parameter
objects and state-dict names stay intact. Failed setup restores methods already
changed. The returned controller's `close()` restores the instance explicitly.

This is a deliberately broader eager boundary than the final research policy.
**Graph breaks are intentional, and the extra Python dispatch may erase compiler
gains.** The opaque C++ operator policy from that research has not been ported
here. These draft controls establish an implementation boundary, not numerical
qualification or a serving speedup on the current stack.

Static compilation disables pattern/prologue/epilogue fusion and compiler-owned
CUDA graphs. Each core call scopes Dynamo's 64-specialization / 256-accumulated
budget, hard failure at exhaustion, and error suppression off. The previous
process settings are restored on both success and error. There is no eager retry
after a compiled error; restart the worker. Calls within a worker must be serial.
`runtime._vtsfm_compiled_forecast.describe()` reports the selected policy and
successful core calls; it is not proof that every operation compiled. The manual
gate separately checks captured-graph activity. It expects native eager islands.

## Validation before enabling outside a draft

CPU tests cover discovery, launch isolation, unsupported combinations, all three
target selections, source-change rejection, method rollback, and compiler-budget
lifetime/error propagation. They use a fake compiler and do not establish CUDA
numerical agreement.

On an idle GPU with each pinned adapter environment, run the manual regression:

```bash
VTSFM_TEST_ADAPTER=moirai2 VTSFM_RUN_COMPILED_FORECAST=1 \
  TRITON_DEFAULT_FP_FUSION=0 NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=src:tests \
  .venv-moirai2/bin/python -m pytest tests/test_compiled_forecast_runtime.py -q -s
```

Repeat for `timesfm25` and `toto2` with their environments. The test captures
pristine references first, then checks small forecasts, the known B16/C2048/H800
recursive corner, changed values at a retained shape, A/B/A reuse and unchanged
inputs, using `rtol=atol=1e-4`. It checks actual compiler graph capture separately.
This gate has not been run as part of the initial draft.

The remaining #17 work is current-stack all-model parity, retained-worker and
HTTP integration, a maintained opaque native-operator backend, explicit CUDA
graph replay/buffer ownership, and independent performance/accuracy evaluation.
Use the existing manual model profiles and preserve raw evidence. Human-run
profiling and speed measurements remain separate from this implementation PR.
