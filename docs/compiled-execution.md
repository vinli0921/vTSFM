# Compiled forecasts

TimesFM, Toto and Moirai can compile their neural cores with `torch.compile`:

```bash
./scripts/serve.sh --adapter toto2 --forecast-execution compile
```

Eager remains the default. `VTSFM_FORECAST_EXECUTION` accepts `eager` or `compile`;
an explicit CLI value takes precedence. Restart the worker to change modes.
The option applies to the vTSFM numerical adapters. CLI help, discovery and
dry-runs stay CPU-safe, and vLLM retains its existing `--enforce-eager` setting.

## Execution

Each adapter chooses its core and the calls that must retain native arithmetic.
The shared helper compiles the bound `forward` and uses instance-local
`torch.compiler.disable` boundaries. Modules, parameters, checkpoint keys and
strict weight loading are preserved.

| Adapter | Compiled forward | Native numerical calls |
| --- | --- | --- |
| TimesFM | `model.forward` | RMSNorm, activations, attention's SDPA callable |
| Toto | `model.transformer.forward` | RMSNorm, attention and FFN forwards |
| Moirai | `model.encoder.forward` | RMSNorm, attention forwards, FFN activation |

Padding, normalization, decoding and quantile recursion use the existing tensor
paths. Moirai resolves static `QueryKeyProjection` shape properties before
tracing. For BF16/FP16 Toto, an instance-local layer forward also retains native
unit-scaled residual addition: compiling that scale/add sequence changes
intermediate rounding. The body follows the pinned upstream implementation;
FP32/TF32 keep the original layer forward. Precision selection is separate (#21).

The launcher sets `TRITON_DEFAULT_FP_FUSION=0` before Torch/Triton imports.
Direct runtime users must set it before importing Torch. Compilation uses
`dynamic=False`, `fullgraph=False`, and disables pattern matching, prologue and
epilogue fusion to preserve recursive forecast parity. These eager boundaries
can limit the compiled work. Inductor CUDA Graphs are disabled; explicit graph
capture and replay are not implemented.

The worker sets Dynamo's per-frame/accumulated recompilation limits to 64/256.
TimesFM's decode phases exceeded the default eight specializations. When a legal
new shape exhausts the cache, normal Dynamo warnings and eager fallback keep it
serviceable; retained compiled entries remain usable. Other compiler/model errors
propagate. No shape bucketing or eviction policy is added. See the pinned
[Torch Dynamo settings](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_dynamo/config.py)
and [Inductor settings](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/config.py).

## Validation

Use the existing manual profiles on an idle GPU:

```bash
./scripts/test-models.sh --adapter all --forecast-execution compile
./scripts/test-models.sh --adapter all --profile accuracy --forecast-execution compile
./scripts/test-models.sh --adapter all --profile performance --forecast-execution compile
```

The native reference and native HTTP comparator stay eager. Artifacts record each
backend's execution mode and the server's contraction setting. Untimed parity
requests allow 900 seconds for a cold compiled shape; timed requests retain their
60-second limit. Compare against a matching eager vTSFM run to isolate compilation
from the rest of the serving stack. Keep startup and warmups outside timing.

Compile integration also runs `tests/test_compiled_forecast_runtime.py`: full
FP32 checkpoints, all nine quantiles, B1/C512/H96 and B16/C2048/H800, changed
values and A/B/changed-B/B/A shape reuse. It checks unchanged inputs, strict
`rtol=atol=1e-4`, and actual compiler activity. This retained-shape check is FP32
regardless of the HTTP profile's selected precision. Tiny official CPU models in
`tests/test_compilation_models.py` additionally check parameter ownership,
checkpoint loading and independent model instances.

On the pinned Torch 2.14/CUDA 13/vLLM 0.33 A100 stack, all three adapters passed
six native-versus-HTTP cases plus the retained-shape check. BF16 plus compilation
also passed the six HTTP cases for Toto and Moirai. These used authorized shared
GPU capacity for correctness; they establish no latency or throughput gain.
Batched serving, TimesFM block scheduling and precision combinations need their
own qualification before attributing a speedup. Issue #17's streaming latency
and historical Chronos acceptance remain open.
