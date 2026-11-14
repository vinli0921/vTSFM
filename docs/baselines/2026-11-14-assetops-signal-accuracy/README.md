# Forecast accuracy on more AssetOpsBench signals — November 14, 2026

Issue #23. Harness: [AssetOps signals](../../model-integration.md#assetops-signals). Per-row data: [summary.csv](summary.csv).

A100-SXM4-40GB, driver 595.91.07. vLLM 0.33.0 V1, Torch 2.14.0+cu130, Transformers 5.20.2, eager FP32,
TF32 off. vTSFM `df22317` (clean tree). Pinned checkpoints: TimesFM `1d95242`, Toto `a7bab28`, Moirai `30f43ff`.
AssetOpsBench data at `e844f64`, pinned in `benchmarks/assetops_sources.json`.

Five signals, one case each: five forecast origins with context 512 and horizon 96, all nine quantiles, chosen
where the whole horizon was recorded. Missing history slots are causally forward-filled; no filled value is
scored. Scores below are the official native models'. Chronos-2 (separate legacy environment) is not included.

```bash
.venv/bin/python -m benchmarks.prepare_data --output data
.venv/bin/python -m benchmarks.model_evaluation prepare --data-dir data \
  --assetops-signals --output data/accuracy-signals.json
./scripts/test-models.sh --adapter all --profile accuracy \
  --cases data/accuracy-signals.json --output results/issue23/fp32-run2
```

## Results (FP32)

| Signal | Seasonality | TimesFM MASE / SQL | Toto MASE / SQL | Moirai MASE / SQL | Seasonal naive MASE | Worst ÷ best MASE / SQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chiller 6 supply temperature | 96 | 0.805 / 0.591 | **0.803 / 0.586** | 0.809 / 0.601 | 1.080 | 1.01 / 1.02 |
| Chiller 6 condenser water flow | 96 | 0.336 / 0.293 | **0.335 / 0.284** | 0.339 / 0.291 | 0.486 | 1.01 / 1.03 |
| MetroPT oil temperature | 24 | **0.663 / 0.522** | 0.879 / 0.634 | 0.758 / 0.561 | 0.887 | 1.33 / 1.21 |
| MetroPT motor current | 24 | 0.987 / **0.844** | 1.122 / 0.917 | **0.969** / 0.853 | 1.211 | 1.16 / 1.09 |
| Hydraulic PS1 pressure | 1 | 7.262 / 5.686 | **6.451 / 4.352** | 9.265 / 7.369 | 11.696 | 1.44 / 1.69 |

Bold marks the best model per metric. SQL is the harness's quantile approximation of a scaled CRPS; seasonal
naive is a point forecast, so its SQL equals its MASE. Hydraulic values are one per machine cycle with
seasonality 1, so its MASE scale is the one-cycle change and values above 1 are expected; compare models with
each other and with naive there.

## Answer

- **"Accuracy is a tie" holds only for the chiller signals.** On both chiller signals all three models are
  within 1–3%. On the others they are not: worst ÷ best MASE is 1.33 for oil temperature, 1.16 for motor current
  and 1.44 for hydraulic pressure (1.69 in SQL). No model is best everywhere: Toto leads on the chiller and
  hydraulic signals, TimesFM on oil temperature, and Moirai/TimesFM split motor current.
- **Every model beats seasonal naive on every signal** in MASE and SQL. The narrowest margin is Toto on oil
  temperature (0.879 vs 0.887).
- **"BF16 is lossless" is not tested yet.** Native BF16 (#21, PR #57) is not on `main`; it merged into the
  compilation branch (#56). Combine an FP32 and a BF16 run with `python -m benchmarks.accuracy_table`, which then
  adds per-signal BF16 − FP32 deltas.

## Parity

TimesFM matched native exactly on all 21,600 values. Toto and Moirai passed every signal except condenser water
flow, where 79 (Toto) and 68 (Moirai) of 4,320 values exceed the `rtol=atol=1e-4` gate. All failures are
forecasts within 9 of zero in series whose forecasts reach about 10,000. Measured in FP32 steps at each series'
largest value, the largest mismatch is 5 (Toto) and 12.5 (Moirai) on condenser flow, against 2–9.5 on the
signals that pass. The served paths therefore differ from native by a few rounding steps everywhere; near zero,
a series in the thousands cannot be resolved to the gate's 1e-4 absolute floor in FP32. The tolerance was not
changed. A floor that scales with series magnitude would be a separate, explicit decision.

## Limits

- One asset per signal and five correlated windows each: evidence about these signals, not a model ranking.
  Differences of a few percent are within what different windows could change.
- One FP32 run; no repeats.
- Chronos-2 and BF16/TF32 are not covered.
