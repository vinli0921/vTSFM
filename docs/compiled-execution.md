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

| Adapter | Compiled forward | Native boundaries and the behavior they preserve |
| --- | --- | --- |
| TimesFM | `model.forward` | RMSNorm and activation forwards preserve eager rounding across normalization and nonlinearities. The attention SDPA callable retains upstream kernel dispatch. Attention projections remain available for tracing. |
| Toto | `model.transformer.forward` | RMSNorm and the gated FFN preserve upstream normalization, activation, and gating arithmetic. The full attention forward retains upstream tensor layout, KV cache handling, and SDPA backend selection, including the BF16 guard documented under [forecast precision](precision.md). |
| Moirai | `model.encoder.forward` | RMSNorm and FFN activations preserve native rounding. The full grouped-query attention forward retains upstream projection, rotary embedding, mask and bias assembly, and SDPA dispatch. |

Padding, normalization, decoding and quantile recursion use the existing tensor
paths. Moirai resolves static `QueryKeyProjection` shape properties before
tracing. For BF16/FP16 Toto, an instance-local layer forward also retains native
unit-scaled residual addition: compiling that scale/add sequence changes
intermediate rounding. The body follows the pinned upstream implementation;
FP32/TF32 keep the original layer forward. This branch includes the
[precision controls](precision.md) from #57.

The preserved behavior explains where the current implementation breaks the
compiled graph. Earlier parity checks exercised this set of boundaries together.
They did not establish which individual boundaries are necessary. To test a
boundary independently, remove only that boundary in a clean experiment checkout,
run the adapter's compiled integration profile, and keep the error statistics
and compiler logs (`TORCH_LOGS=graph_breaks,recompiles`). Keep precision,
checkpoint, contraction and Inductor settings fixed across the comparison.

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

## Moirai attention and precision

| Attention | Eager FP32 | Eager BF16 / TF32 | Compile FP32 / BF16 / TF32 |
| --- | --- | --- | --- |
| `upstream` (default) | Enabled | Enabled | Enabled |
| `univariate` | Enabled | Rejected at startup | Rejected at startup |

The table describes accepted settings. Earlier GPU checks covered Moirai with
upstream attention in eager FP32, BF16 and TF32 (#57), and in compiled FP32 and
BF16 (#56). #46 qualified univariate attention in eager FP32. This integration
changes the startup checks, so rerun parity for the enabled settings before
claiming qualification on the combined revision. Compiled TF32 also needs
separate qualification.

For `VTSFM_MOIRAI_ATTENTION=univariate`, the launcher and direct runtime select
FP32 when precision is unspecified. The launcher sets `NVIDIA_TF32_OVERRIDE=0`
in the child. Direct runtime callers must use an unset or zero override. Moirai
validates the combination before constructing the model. It then configures
precision, installs any selected attention patch, and applies compilation when
enabled. Both univariate attention and compilation install instance-level
attention forwards, so the unqualified combination fails with a configuration
error before either patch is attempted.

## Validation

Use the existing manual profiles on an idle GPU:

```bash
./scripts/test-models.sh --adapter all --forecast-execution compile
./scripts/test-models.sh --adapter all --profile accuracy --forecast-execution compile
```

The native reference and native HTTP comparator stay eager. Artifacts record each
backend's execution mode and the server's contraction setting. Untimed parity
requests allow 900 seconds for a cold compiled shape; timed requests retain their
60-second limit.

Compile integration also runs `tests/test_compiled_forecast_runtime.py`: full
FP32 checkpoints, all nine quantiles, B1/C512/H96 and B16/C2048/H800, changed
values and A/B/changed-B/B/A shape reuse. It checks unchanged inputs, strict
`rtol=atol=1e-4`, and actual compiler activity. This retained-shape check is FP32
regardless of the HTTP profile's selected precision. Tiny official CPU models in
`tests/test_compilation_models.py` additionally check parameter ownership,
checkpoint loading and independent model instances.

On the pinned Torch 2.14/CUDA 13/vLLM 0.33 A100 stack, prior runs passed six
native-versus-HTTP cases plus the FP32 retained-shape check for all three
adapters. BF16 plus compilation passed six HTTP cases for Toto and Moirai.
These runs predate the merge with #46 and later changes on `main`. TimesFM's
optional block scheduler uses the model forward for prefill and its own
incremental decode path. Whole-forecast results cannot qualify that block path.
Issue #17's streaming latency and historical Chronos acceptance remain open.

The selected adapter's [numerical admission limits](numerical-admission.md)
also apply to compiled and reduced-precision requests. The published Toto and
Moirai peak-memory measurements qualify eager FP32 with upstream attention and
TF32 disabled. Recheck the maximum accepted requests and repeated concurrent
waves before relying on those measurements with compilation or reduced precision.

## Matched performance and profiling

On one idle GPU, use the same clean checkout and pinned data for both vTSFM
modes. This compares compiled and eager vTSFM HTTP points at the same precision,
request payload, worker settings and concurrency. The native HTTP path is an
additional control. Run the four passes sequentially and keep separate output
folders:

```bash
export VTSFM_MOIRAI_ATTENTION=upstream TRITON_DEFAULT_FP_FUSION=0
out="results/compile-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
compare() {
  ./scripts/test-models.sh --adapter all --profile performance --gpu 0 \
    --forecast-precision fp32 --forecast-execution "$1" --backend-order "$2" \
    --batches 1,4 --contexts 128,512,2048 --horizon 96 --clients 1 \
    --warmups 10 --requests 100 --output "$out/$3"
}
compare eager   native,vtsfm eager-1
compare compile native,vtsfm compile-1
compare compile vtsfm,native compile-2
compare eager   vtsfm,native eager-2
```

From each `performance.json`, compare successful vTSFM points for p50 and p95
HTTP latency, series/s, and sampled GPU residency. Report cold compilation
separately from the warmed samples. Retain failed points, raw samples, software
and checkpoint revisions, and hardware metadata. Repeat the same comparison
with `--adapter toto2 --forecast-precision bf16` or
`--adapter moirai2 --forecast-precision bf16` for a BF16 claim.

Trace the runtime separately so Nsight overhead stays outside the HTTP latency
comparison. On the same idle GPU, set `NSYS`, `AOB_ROOT`, and `HF_HOME` as
in the [Nsight harness guide](../benchmarks/nsight/README.md). For each adapter,
run this trace in both modes with the same context, horizon and warmup count:

```bash
export PYTHONPATH="$PWD/src" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0 VTSFM_PRECISION=fp32 OMP_NUM_THREADS=4
for adapter in timesfm25 toto2 moirai2; do
  py=".venv-$adapter/bin/python"
  for mode in eager compile; do
    trace="$out/$adapter-runtime-$mode"
    VTSFM_FORECAST_EXECUTION="$mode" "$NSYS" profile --trace=cuda,nvtx,osrt \
      --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o "$trace" "$py" benchmarks/nsight/driver.py --model "$adapter" --path runtime \
      --round 1 --warm 10 --n 50 --out "$trace.json"
    "$NSYS" export --type sqlite -o "$trace.sqlite" "$trace.nsys-rep"
    "$py" benchmarks/nsight/analyze.py "$trace.sqlite" > "$trace-analysis.jsonl"
  done
done
```

Compare CUDA launches and GPU busy time per request alongside the untraced HTTP
measurements. Review graph breaks and recompilations in a separate diagnostic run
with `TORCH_LOGS=graph_breaks,recompiles`. No matched profile at this revision
has established the size or direction of a speed change.
