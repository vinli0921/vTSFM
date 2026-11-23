# Forecast accuracy on AssetOpsBench signals, non-overlapping windows — November 23, 2026

This baseline replaces the [November 14 baseline](../2026-11-14-assetops-signal-accuracy/README.md)
for ranking claims ([#84](https://github.com/vinli0921/vTSFM/issues/84)). Across all windows, no model can
be ranked above another on any signal. The mean of the recorded history beats all three models on MetroPT
motor current, and on condenser flow both baselines score worse than the best model. Read rankings only
from the paired comparison in
[`accuracy_table.md`](accuracy_table.md), not from the spread of mean MASE.

A100-SXM4-40GB, driver 595.91.07. vLLM 0.33.0 V1, Torch 2.14.0+cu130, Transformers 5.20.2, eager FP32,
TF32 off. vTSFM `8518a0e` (clean tree). Pinned checkpoints are unchanged from the November 14 baseline.
AssetOpsBench data at `e844f64`, pinned in `benchmarks/assetops_sources.json`. The prepared manifest
`accuracy-signals-nonoverlap.json` had SHA-256
`65a38bc965b49f53943033353bc96ebf975097cda07a57e1731591a2aa3785ae`.

Each signal uses every forecast origin with 512 history steps and 96 recorded future values whose horizon
does not overlap another chosen origin: 24 windows for each Chiller 6 signal, 17 for Hydraulic PS1 and 11
for each MetroPT signal. At C512/H96, MetroPT and Hydraulic cannot reach 20 such windows. Windows from one
asset are correlated, so the bootstrap intervals are approximate. Scores use the official native FP32
forecasts. MAE and MASE use the median; SQL uses all nine quantiles.

| File | Contents |
| --- | --- |
| [`accuracy_table.md`](accuracy_table.md) | Summary, paired comparison with 95% intervals, and model spread |
| [`summary.csv`](summary.csv) | One row per model and signal, including both robust aggregates and both baselines |
| [`comparison.csv`](comparison.csv) | Paired-bootstrap comparison rows, with and without off-state windows |
| [`windows.csv`](windows.csv) | One row per model, signal and window, with flags and baseline scores |

## What the paired comparison supports

- **MetroPT motor current (11 windows):** the history mean is lowest (ratio of means 0.934). TimesFM,
  Toto, Moirai and seasonal naive are all higher, each at least 1.43× the lowest with intervals above 1.
- **Chiller 6 condenser flow:** 19 of 24 windows are off-state (at least half the future labels are
  exactly zero). Across all 24, the models cannot be told apart, and seasonal naive (1.46×) and the
  history mean (1.79×) score higher than the lowest model, Moirai. On the 5 on-state windows TimesFM is
  lowest, but Toto's ratio interval is [1.00, 1.01]; treat that as fragile at the 5-window minimum.
- **Hydraulic PS1, MetroPT oil temperature:** no model, or seasonal naive, can be told apart from the lowest.
- **Chiller 6 supply temperature:** flagged unforecastable (lag-1 autocorrelation −0.01 to 0.12); scored
  but excluded from ranking claims.

## Parity

All three adapters fail the `rtol=atol=1e-4` native parity gate on 3 of 13 accuracy cases, all on
condenser water flow (largest absolute error 0.0068 on a series reaching about 10,000). Every other case
passes. TimesFM matched native exactly with the November 14 five-window cases, and the CPU-built
rotary-table fix ([#90](https://github.com/vinli0921/vTSFM/pull/90)) applies only to Toto and Moirai, so
TimesFM's failure here needs the magnitude-aware floor tracked in
[#71](https://github.com/vinli0921/vTSFM/issues/71). The table reports native scores, so the failures do
not change the accuracy numbers.

```bash
.venv/bin/python -m benchmarks.prepare_data --output data --assetops-signals
.venv/bin/python -m benchmarks.model_evaluation prepare --data-dir data \
  --assetops-signals --output data/accuracy-signals-nonoverlap.json
./scripts/test-models.sh --adapter all --profile accuracy \
  --cases data/accuracy-signals-nonoverlap.json --output results/issue84/fp32
.venv/bin/python -m benchmarks.accuracy_table results/issue84/fp32 --output results/issue84/fp32
```
