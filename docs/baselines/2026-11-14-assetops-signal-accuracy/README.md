# Forecast accuracy on more AssetOpsBench signals — November 14, 2026

> **Superseded for ranking claims ([#84](https://github.com/vinli0921/vTSFM/issues/84)).** This baseline
> scores five correlated windows per signal with the mean of per-window MASE ratios, which one tiny-scale
> window can dominate. Do not read per-signal winners or "models differ by 16–44%" from it. The accuracy
> harness now selects every non-overlapping window (24/24/11/11/17 at C512/H96), adds a mean-of-history
> baseline, ratio-of-means and geometric-mean aggregates, per-window ranks, paired-bootstrap intervals and
> off-state/unforecastable flags; see [the accuracy profile](../../model-integration.md#assetops-signals).
> The [November 23 baseline](../2026-11-23-assetops-signal-accuracy-nonoverlap/README.md) replaces these comparisons. The commands below used the then-default five
> evenly spaced origins; reproduce them with `--signal-windows 5`. This baseline's history-mean column
> averages all 512 inputs, including forward-filled slots; the new baseline uses recorded values only.

The [accuracy harness](../../model-integration.md#assetops-signals) evaluates five signals for [issue #23](https://github.com/vinli0921/vTSFM/issues/23). The [CSV](summary.csv) contains one row per model and signal.

A100-SXM4-40GB, driver 595.91.07. vLLM 0.33.0 V1, Torch 2.14.0+cu130, Transformers 5.20.2, eager FP32,
TF32 off. vTSFM `df22317` (clean tree). Pinned checkpoints: TimesFM `1d95242`, Toto `a7bab28`, Moirai `30f43ff`.
AssetOpsBench data at `e844f64`, pinned in `benchmarks/assetops_sources.json`.

The November 14 run directory was unavailable when we regenerated the CSV. On November 21 we replayed
the same inputs at the original runtime revision on the A100 and generated the CSV with the table code at
`467f2ff`. All 15 rows exactly reproduced the previously published fields. The new columns record the
scale policy and percentage of forward-filled history. The team archive on the A100 is
`~/vtsfm-pr66-fp32-20261121/pr66-fp32-five-signals-df22317-20261121.tar.gz`
(SHA-256 `ae82bbc4ec262d8c98f008a9c80e32db4cbb6a85d1e587c410b60a99551091c3`).

The five signals come from three assets. For each signal we selected five forecast origins with 512 history
steps and 96 recorded future values. Missing history slots were causally forward-filled. The scores below use
the official native FP32 forecasts and give equal weight to each origin. MAE and MASE use the median forecast.
Scaled quantile loss (SQL) summarizes all nine quantiles.

```bash
.venv/bin/python -m benchmarks.prepare_data --output data --assetops-signals
.venv/bin/python -m benchmarks.model_evaluation prepare --data-dir data \
  --assetops-signals --signal-windows 5 --output data/accuracy-signals.json
./scripts/test-models.sh --adapter all --profile accuracy \
  --cases data/accuracy-signals.json --output results/issue23/fp32-run2
```

To regenerate the published CSV from a completed run directory with this PR's table generator
(`benchmarks/accuracy_table.py` at `467f2ff`; later generators append columns and window/comparison CSVs):

```bash
run=results/issue23/fp32-run2
.venv/bin/python -m benchmarks.accuracy_table "$run" --scale all --output "$run"
cp "$run/accuracy_table.csv" docs/baselines/2026-11-14-assetops-signal-accuracy/summary.csv
```

## Results (FP32)

| Signal | Seasonality | TimesFM MASE / SQL | Toto MASE / SQL | Moirai MASE / SQL | Seasonal naive MASE | History mean MASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chiller 6 supply temperature | 96 | 0.805 / 0.591 | 0.803 / 0.586 | 0.809 / 0.601 | 1.080 | 0.806 |
| Chiller 6 condenser water flow | 96 | 0.336 / 0.293 | 0.335 / 0.284 | 0.339 / 0.291 | 0.486 | 0.767 |
| MetroPT oil temperature | 24 | 0.663 / 0.522 | 0.879 / 0.634 | 0.758 / 0.561 | 0.887 | 0.736 |
| MetroPT motor current | 24 | 0.987 / 0.844 | 1.122 / 0.917 | 0.969 / 0.853 | 1.211 | 0.835 |
| Hydraulic PS1 pressure | 1 | 7.262 / 5.686 | 6.451 / 4.352 | 9.265 / 7.369 | 11.696 | 14.197 |

The history-mean baseline predicts the arithmetic mean of the same 512 input values at each of the 96
future steps. Both baselines are point forecasts, so their SQL equals their MASE when repeated across the
nine quantiles.

### Sensitivity to the forecast origin

- **Chiller 6 supply temperature.** Lag-1 autocorrelations for the five input histories range from −0.012
  to 0.124. Predicting the history mean gives MASE 0.806, within the models’ 0.803–0.809 range. On these
  windows, the three models had about the same median-forecast error as predicting a constant.
  Across 480 predictions per model, the native medians differed from their own window’s history mean
  by 5.40 signal units on average for TimesFM, 4.58 for Toto and 6.72 for Moirai (absolute differences).
  The actual future values differed from that mean by 33.38 units on average.
- **Chiller 6 condenser flow.** At origin 512, all 96 future readings exceed 250 and reach 11,030.
  At each of the four later origins, 79–85% of future readings are zero and the maximum is below 10.
  The first origin accounts for 99.7–99.8% of each model's total absolute error. Two of the near-zero
  windows have large seasonal scales (3,498 and 366), which give even smaller scaled errors for those windows.
- **MetroPT.** Predicting the history mean gives lower MASE than all three models for motor current (0.835),
  and lower MASE than Toto and Moirai for oil temperature (0.736). All three models score below seasonal
  naive on the sampled windows. For Toto on oil temperature the margin is 0.879 versus 0.887.
- **Hydraulic PS1.** Each window’s scale is its mean absolute one-cycle training change. The five scales
  are 1.479, 2.489, 0.026, 0.027 and 0.030 bar. A given absolute error therefore contributes much more to
  MASE in the last three windows. Mean MAE in bar is 1.506 for Moirai, 1.516 for TimesFM and 1.653 for
  Toto, the reverse of their order under mean MASE. A small scale can push MASE above one.

### MetroPT history imputation

Oil temperature and motor current use the same five forecast origins. In each 512-slot history,
32–89 slots (6–17%) were forward-filled. The reported seasonal scale uses all 488 lag-24 pairs
from each history. Restricting the scale calculation to pairs with two observed readings gives
scales 3–12% smaller for oil temperature and 4–31% smaller for motor current across these
origins. Each model saw the same forward-filled inputs.

| Forecast origin slot | Forward-filled history slots | Lag-24 pairs with both readings observed |
| ---: | ---: | ---: |
| 559 | 32 | 435 |
| 3177 | 62 | 370 |
| 3409 | 39 | 412 |
| 4015 | 89 | 331 |
| 4534 | 51 | 391 |

Re-scoring with scales over recorded pairs only (`accuracy_table --scale observed`, an exact rescaling of
this run's scores) raises the sampled MetroPT MASE because the scales shrink.

| Signal | TimesFM MASE | Toto MASE | Moirai MASE | Seasonal naive MASE | Largest ÷ smallest sampled MASE / SQL |
| --- | ---: | ---: | ---: | ---: | ---: |
| MetroPT oil temperature | 0.711 | 0.945 | 0.818 | 0.956 | 1.33 / 1.22 |
| MetroPT motor current | 1.200 | 1.321 | 1.169 | 1.463 | 1.13 / 1.07 |

The order of the three mean MASE values is the same under both scale policies on these five origins.
Toto's oil-temperature margin over seasonal naive narrows to 0.945 versus 0.956. Chiller histories are
0.2% filled and hydraulic histories are fully observed.

## Interpretation

Each signal represents one asset and five correlated origins. Changing the origin, error scale or baseline
can change the comparison. Broader sampling and aggregation across operating regimes are tracked in [#84](https://github.com/vinli0921/vTSFM/issues/84).

This run covers FP32. A matched BF16/TF32 accuracy comparison on these signals remains part of
[#23](https://github.com/vinli0921/vTSFM/issues/23). The table generator can combine runs made with the
same inputs at different precisions.

## Parity

TimesFM matched native exactly on all 21,600 values. Toto and Moirai passed every signal except condenser water
flow, where 79 (Toto) and 68 (Moirai) of 4,320 values exceed the `rtol=atol=1e-4` gate. All failures are
forecasts within 9 of zero in series whose forecasts reach about 10,000. Measured in FP32 steps at each series'
largest value, the largest mismatch is 5 (Toto) and 12.5 (Moirai) on condenser flow, against 2–9.5 on the
signals that pass. The relative tolerance contributes little near zero, so those differences can exceed
the gate's absolute tolerance. The accuracy profile exits nonzero on these failures.

Follow-up work ([#71](https://github.com/vinli0921/vTSFM/issues/71)) should measure a magnitude-aware absolute floor on the reference GPU using this condenser-flow
case and the passing signals. Include perturbed forecasts to verify that the candidate still rejects material
deviations. The accuracy profile continues to require `rtol=atol=1e-4`.

The replay uses the same five origins per signal. This evaluation covers TimesFM, Toto and Moirai.
