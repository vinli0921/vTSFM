# Forecast precision

Precision is opt-in for a worker's lifetime. Without `--forecast-precision`,
serving keeps its existing math settings and model dtypes. The FP32 numerical
request/result contract and pinned checkpoints are unchanged.

```bash
./scripts/serve.sh --adapter toto2 --forecast-precision bf16
./scripts/serve.sh --adapter moirai2 --forecast-precision bf16
./scripts/serve.sh --adapter timesfm25 --forecast-precision tf32
```

Modes:

- `fp32`: highest FP32 matmul precision, with TF32 disabled.
- `tf32`: FP32 weights and tensors, with high matmul precision and TF32 enabled.
- `bf16`: native BF16 compute weights for Toto and Moirai, without autocast.

The launcher reads `VTSFM_PRECISION`; an explicit CLI selection overrides it.
For a selection it sets the matching `NVIDIA_TF32_OVERRIDE` before child startup.
With neither flag nor variable, existing math settings are preserved. Direct
runtime construction also reads `VTSFM_PRECISION`; contradictory CUDA overrides
produce an error before execution.
Torch's matmul setting is process-global, so use separate workers for different
modes. CLI discovery, help and dry-runs stay CPU-safe.

## Native BF16

Each adapter selects its existing input/output projections, normalization and
rotary modules. A shared conversion helper casts compute weights once, preserving
exact FP32 norm parameters and floating numerical buffers. Instance-local hooks
run normalization and RoPE in FP32, then return BF16 states to the transformer.
Module names and strict checkpoint loading remain unchanged.

Explicit BF16 and TF32 workers use the upstream CPU rotary table construction,
then move the FP32 tables to the model device. This matches the official
checkpoint loaders even when vLLM constructs the model directly on CUDA.
CPU/CUDA sin and cos can differ by one FP32 ulp; a later BF16 rounding can
amplify that difference during recursive forecasting. Configured frequencies
and table lengths are preserved. The default FP32 path is unchanged.

Inputs, scaling and forecast feedback remain FP32. Upstream scalers retain their
own FP64 statistics. The output projection returns FP32 before inverse transforms,
Toto's sinh/sorting/median feedback and Moirai's recursive nine-trajectory
quantile reduction. SDPA retains its native accumulation behavior. Moirai uses
the tensor forecast path from #10; no forecast loop or attention kernel is copied.

Moirai BF16 and TF32 group rows by their original history length, matching the native
reference's batch geometry. Mixed padded batches can otherwise change CUDA
BF16/TF32 rounding. Outputs retain their original row order and histories stay on
the device. The live tensor entrypoint synchronizes only integer length metadata;
prepared execution reuses the known lengths. FP32 batching is unchanged.

For Toto BF16, singleton variate attention with more than 65,535 flattened
CUDA batches excludes FlashAttention locally, allowing efficient attention and
the upstream math backend. All other calls retain their original selection.
The instance-local wrapper calls the original attention forward and restores
backend flags, including after errors. At B256/C8192/H96 this attention batch is
256 × (256 context patches + 3 forecast patches) = 66,304; the pinned Flash
launcher exceeds its CUDA grid limit. The matching BF16 native reference records
and uses this same explicit backend policy because the original full-shape
kernel fails. The preserved original failure is not a parity pass.

TimesFM native BF16 is outside #21; it is rejected by this option. TF32 keeps the
existing FP32 transport/cache layout and is not prohibited with TimesFM block
scheduling. That combination still needs its own GPU qualification.

## Manual checks

The existing manual runner forwards precision to both official native inference
and vTSFM HTTP, and records the mode in runtime and performance artifacts:

```bash
./scripts/test-models.sh --adapter moirai2 --forecast-precision bf16 --profile integration
./scripts/test-models.sh --adapter toto2 --forecast-precision bf16 --profile accuracy
./scripts/test-models.sh --adapter all --forecast-precision tf32 --profile performance
```

The runner's default remains the existing FP32/TF32-off control. Run FP32 and BF16
accuracy profiles separately and compare observed-label scores; parity against a
matching BF16 native reference does not establish lossless precision. Native and
served models share the dtype conversion policy but execute separate official
forecast paths. CPU tests independently inspect conversion values and boundaries.

On the pinned Torch 2.14/CUDA 13/vLLM 0.33 A100 stack, the six native-versus-HTTP
cases passed for Toto/Moirai BF16 and all three TF32 adapters, including long,
ragged and repeated forecasts. Tiny official CPU/CUDA models additionally check
conversion values, sensitive dtype boundaries, checkpoint keys and instance
isolation.

Separate FP32 and reduced-precision accuracy profiles scored 1,120 observed labels
per model/mode from five correlated AssetOps sensor windows. Mean MAE changed by
+0.1975% for Toto BF16 and +0.0429% for Moirai BF16; MAE/MASE/SQL changes for TF32
were below 0.033% on this slice. These are limited regression checks, not a
lossless-precision or general model-accuracy claim.

A separate synthetic Toto BF16 B256/C8192/H96 check compared all 221,184 values
bitexact against native inference using the documented non-Flash policy. The
original native Flash failure is retained separately. This checks the runtime,
not the HTTP request envelope or concurrent admission limits.

GPU correctness checks used authorized shared capacity. Latency, throughput and
comparative memory remain manual acceptance work for #21. Preserve strict parity
tolerances and compare matching FP32, TF32 and BF16 workloads on an idle GPU.
