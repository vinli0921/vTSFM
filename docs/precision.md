# Experimental forecast precision

The default stays FP32 with highest matmul precision. An explicit worker option
starts a separate precision experiment without changing checkpoint revisions,
HTTP inputs, quantile selection, or the FP32 result-envelope format:

```bash
./scripts/serve.sh --adapter toto2 --forecast-precision bf16
./scripts/serve.sh --adapter toto2 --forecast-precision tf32
```

`--forecast-precision fp32|tf32|bf16` is CPU-safe to inspect with `--dry-run`.
The launcher passes `VTSFM_PRECISION` explicitly, replacing any inherited value.
The selection is fixed for the worker lifetime; it is not a per-request option.
Do not combine models with different precision settings in one worker process.
For numerical adapters, the launcher also sets `NVIDIA_TF32_OVERRIDE` before
child startup (`1` for TF32, `0` otherwise), so an inherited benchmark override
cannot silently change the requested mode. Direct/offline runtime construction
rejects conflicting overrides; use a fresh process with matching settings.
Historical Chronos launch settings are unchanged.

## Supported draft paths

* `fp32`: existing TimesFM/Toto/Moirai forecasts, highest FP32 matmul precision.
* `tf32`: the same numerical adapters with Torch's `high` FP32 matmul setting and
  CUDA TF32 matmuls enabled. Dtypes and model weights stay FP32.
* `bf16`: native Toto compute weights, without autocast. Norms retain the pinned
  FP32 arithmetic and parameters; RoPE frequency tables retain their exact FP32
  values, along with the unit-scaling residual constants. Patch-projection
  inputs become BF16 after FP32 scaling. The output
  head returns FP32 before sinh, inverse scaling, quantile sorting and feedback.
  The official causal scaler still uses its FP64 statistics. Torch SDPA owns its
  internal accumulation; this draft adds no attention kernel or softmax rewrite.

Logical input/output dtype in `BatchKey` and `ExecutionResult` remains FP32. BF16
describes internal model compute, not a new public request dtype. Checkpoint
parameter names are unchanged; loading follows the ordinary strict weight path.
Precision hooks are attached only to that Toto instance, before cache creation.
The launch command retains `--dtype float32` for the numerical transport contract.

Native BF16 for Moirai is pending its tensor-path integration and sensitive
operation audit. TimesFM BF16 and historical Chronos precision changes are also
outside this first draft. Unsupported selections fail explicitly. FP16/autocast
are not aliases for native BF16, and BF16 does not turn on TF32 for FP32 work.
Forecast-block continuous batching remains FP32-only; both the CLI and direct
runtime setup reject combining it with TF32/BF16 in this draft.

## Validation remaining before acceptance

This is a draft implementation. Tiny CPU tests check dtype boundaries, exact
retention of sensitive values, instance isolation, checkpoint names and explicit
launch selection. They do not qualify checkpoint forecasts or establish a
speedup. The existing FP32 parity checks and tolerances stay unchanged.

On the pinned Torch 2.14/CUDA 13 GPU environment, independently validate the
selected mode against its matching native reference, including the 767/768/769
and multi-block Toto horizons, masked inputs, changed batch shapes and A/B/A
reuse. Then evaluate observed-label accuracy and memory/performance separately
using the team's manual workflow. The current model-profile runner does not yet
accept this precision option; add explicit precision plumbing before using it
to claim a reduced-precision comparison. Preserve separate FP32 controls and
report metric differences; no lossless-BF16 or speedup claim is made here.
