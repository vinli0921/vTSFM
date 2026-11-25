# Toto native-layout inference scaling on the A100

The original [whole-forecast compilation study](../2026-11-25-toto-whole-compile/README.md)
found strict Torch FP32 parity misses on the Chiller 6 condenser-flow test in
both eager and compiled serving. All misses occurred on the second history in
a B5/C512/H96 request. Its readings ranged from 0 to 11,207 with 144 zeros,
and the missed quantiles were near zero after denormalization.

The direct tensor controls used the pinned checkpoint, an A100-SXM4-40GB,
Torch 2.14.0+cu130 and FP32 with TF32 disabled. Each invocation independently
loaded the official native model and reproduced the archived native forecasts
exactly before measuring its own arm. The eager service arithmetic reproduced
the HTTP mismatch without a server. The tensor-only shim preserves the legacy Python-scalar identity shortcut.
The two layout-aware policies constrain or remove that shortcut.

| Eager control | Values outside tolerance | Max absolute error | Direct p50 (ms) |
| --- | ---: | ---: | ---: |
| Default fast shim | 69 of 4,320 | 0.00390625 | 65.31 |
| Default shim with tensor scales multiplied on device | 69 of 4,320 | 0.00390625 | 61.23 |
| Upstream autograd wrappers | 0 of 4,320 | 0 | 116.20 |
| Fast shim with all forward scales multiplied | 0 of 4,320 | 0 | 76.95 |
| Fast shim with scalar identity skipped only on contiguous inputs | 0 of 4,320 | 0 | 62.57 |

The optimized shim could return a strided input directly for a 1.0 scale.
The upstream multiplication materializes a new tensor with different strides
when the input is a non-dense view. Keeping this operation removed the tested
eager discrepancy. It also avoids reading CUDA tensor scales in a Python
condition. The `native-layout` policy retains this arithmetic while bypassing
`torch.autograd.Function.apply` bookkeeping. The `stride-aware` policy also retains the scalar identity shortcut for
contiguous tensors. Both policies are opt-in. The default uses the legacy
shim, and `VTSFM_TOTO_FAST_UNIT_SCALING=0` uses upstream wrappers.

The direct measurements are diagnostic medians from ten synchronized calls
after warmup. They measure one isolated numerical invocation and exclude HTTP,
preprocessing, and compilation. The strict tolerance was `1e-4+1e-4*abs(native)`. The underlying study
used 100 timed calls for each HTTP timing point.

Broader compilation introduced a separate discrepancy on the same inputs.
The whole-forecast control produced 155 strict misses with 44.47 ms p50 in
the synchronized direct call. Whole-forecast compilation with the tensor-only
scaling change cut p50 to 10.06 ms and still produced 117 misses. The stride-aware
compiled control took 10.37 ms p50 with 145 misses. It took 78.8 s for first
use, compared with 28.9 s in the original whole-compile control. This arithmetic
probe uses a separate measurement boundary from the earlier HTTP comparison.

## End-to-end qualification

Eager serving with `VTSFM_TOTO_FAST_UNIT_SCALING=native-layout` matched the
independent native checkpoint bit for bit across five AssetOpsBench signals.
The five B5/C512/H96 requests compared 21,600 quantile values, including all
4,320 condenser-flow values, and scored 2,400 observed labels. The native and
served accuracy scores were identical. The existing `rtol=atol=1e-4` gate and
checkpoint were used both in the reproducer and the HTTP profile.

## Reproduce the precision ablation

Use the original five-signal cases and native oracle, both produced by
`scripts/test-models.sh --adapter toto2 --profile accuracy --cases MANIFEST`.
Each arm uses an independent process. The results include exact native
fingerprints and counts by row and quantile.

```bash
export PYTHONPATH=src:tests:.
export NVIDIA_TF32_OVERRIDE=0 TRITON_DEFAULT_FP_FUSION=0 HF_HUB_OFFLINE=1
py=.venv-toto2/bin/python
for arm in eager eager-native-scaling eager-tensor-scale eager-native-layout eager-stride-aware \
           whole whole-tensor-scale whole-native-layout whole-stride-aware; do
  "$py" -m benchmarks.toto_compile_ablation --arm "$arm" --cases "$cases" \
    --archived-native "$reference" --output "$out/$arm.json"
done
```
