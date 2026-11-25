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
The all-multiply compiled arm also missed 145 values at 10.69 ms p50. With
tensor scales kept on-device and legacy scalar identity scaling, compiling
only the transformer missed 113 values at 12.00 ms, and compiling the outer
forecast with the transformer eager missed 102 at 61.11 ms.

Two more controls used the corrected stride-aware scaling policy. Keeping the
causal scaler eager while compiling the whole forecast produced 132 misses at
12.35 ms p50. Keeping forecast preparation and output processing eager while
compiling just the full transformer produced 105 misses at 12.18 ms. Restoring
the upstream scaler alone did not resolve the condenser mismatch. The tested
wide scopes remain experimental.

## End-to-end qualification and serving throughput

Two eager inference modes (`native-layout` and `stride-aware`) matched the
independent native checkpoint bit for bit across five AssetOpsBench signals.
Each arm compared 21,600 quantiles and scored 2,400 observed labels on
B5/C512/H96 requests. The original eager path missed 69 of 4,320 condenser-flow
values. A separate eager `stride-aware` integration run matched all 8,586 values
across six cases, including sparse quantiles on ragged inputs, H800 crossing
the native decode-block boundary, repeated shapes, and two MetroPT windows.

Whole-forecast compilation with stride-aware scaling still missed **145 of
4,320 condenser-flow quantiles** over HTTP. The other four signals passed.
The strict, unchanged `rtol=atol=1e-4` check applies to every quantile. Keep the
whole-forecast compile path experimental and use eager when strict correctness
on this workload matters.

| MetroPT workload | Legacy req/s | Stride-aware req/s | Throughput gain | Legacy → stride-aware p95 |
| --- | ---: | ---: | ---: | ---: |
| B1/C2048/H96, one client | 13.41 | 14.07 | 4.9% | 76.10 → 72.85 ms |
| B1/C2048/H96, four clients | 14.57 | 15.42 | 5.8% | 278.47 → 267.37 ms |
| B4/C512/H96, one client | 13.13 | 13.65 | 4.0% | 78.49 → 77.06 ms |
| B4/C512/H96, four clients | 14.54 | 15.15 | 4.2% | 281.68 → 274.72 ms |

The HTTP comparison used four fresh workers in legacy/stride/stride/legacy
order with shape order reversed in the second block. All 1,600 timed requests
completed. The independent native output check ran before each timed shape.
Each cell pools 200 requests after warmup. Throughput divides completions by
summed elapsed wall time. p95 uses the pooled client latencies and includes
queueing at the one serving slot. Both arms peaked at 1,312.63 MiB of worker
Torch allocation for B1 and 1,313.89 MiB for B4, and at 2,416.56 MiB of sampled
device residency. The tracked serving revision was `3ca81ca`. Its untracked
`.venv-toto2` symlink addressed the same pinned environment in all arms.
NVML ownership checks found no foreign GPU processes, and each worker shut
down cleanly.

Set the experimental policy before starting eager FP32 serving.

```bash
VTSFM_TOTO_FAST_UNIT_SCALING=stride-aware ./scripts/serve.sh --adapter toto2 --forecast-precision fp32
```

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
