# Baseline profiling: Chronos-2, TimesFM-2.5, Toto-2, Moirai-2.0

`harness.py` measures and traces four time-series foundation models on one GPU, both
through AssetOpsBench's TSFM engine (`run_recipe`) and through each model's native
pipeline. It is the "before" picture for vTSFM optimizations.

| Model | Checkpoint | Native path | sktime class (run_recipe) |
| --- | --- | --- | --- |
| chronos2 | `amazon/chronos-2` | `Chronos2Pipeline.predict` | `Chronos2Forecaster` |
| timesfm2 | `google/timesfm-2.5-200m-transformers` | `model(past_values=...)` | `TimesFM2Forecaster` |
| toto2 | `Datadog/Toto-2.0-313m` | `Toto2Model.forecast` | `Toto2Forecaster` |
| moirai2 | `Salesforce/moirai-2.0-R-small` | `Moirai2Forecast.forward` | `Moirai2Forecaster` |

## Rounds

| Round | What | Batch | Context | Precision |
| --- | --- | --- | --- | --- |
| 0a | `run_recipe` forecast, zero-shot | 1 | 2048 (+96 holdout) | fp32 |
| 0b | native forward | 64 | 2048 | fp32 |
| 1 | streaming: window advances one model patch per request | 1 | 2048 | fp32 |
| 2 | native forward; on OOM, halve batch until it fits | 64 | 8192 | fp32 |
| 3 | 0b in bf16 | 64 | 2048 | bf16 |
| 4 | `run_recipe` `tsfm_anomaly_detection` (conformal) | 1 | 2048 (+96) | fp32 |
| sweep | timing only | 1, 4, 16, 64, 256 | 512, 2048, 8192 | fp32 |

Each config runs 5 warm-up and 20 timed requests (round 4: 1 and 5, because
conformal AD refits the forecaster many times per request), then `torch.profiler` (CPU
+ CUDA, `record_shapes`) over 3 more and exports a Chrome trace. Seeds are fixed (0).
`preprocess` / `forward` / `postprocess` and the engine's backtest call are marked
with both NVTX ranges (for `nsys`) and `record_function` (visible in the trace).
The horizon is 96 throughout. Forecast quality is MASE (naive scale from the context)
and, where the model returns quantiles, weighted quantile loss as a CRPS estimate.

## Data

`AssetOpsBench/src/couchdb/scenarios_data/shared/iot/chiller_6.json`, column
`Chiller 6 Supply Temperature`, 15-minute samples, about 2.9k points after sorting by
timestamp and dropping empty rows. `Condenser Water Flow` was the first choice, but it
is ~53% zeros (chiller off) and ends in an off period, so its holdout MASE is
meaningless; pass `--target "Chiller 6 Condenser Water Flow"` to use it anyway.
Contexts longer than the series (8192) tile it end to end; such rows have
`context_tiled=true` and their MASE is optimistic.

## Running (Insomnia)

```bash
# once per node: env on node-local disk (home quota is too small)
srun --jobid=<holder> --overlap bash profiling/setup_env.sh
# rounds, each (round, model) in its own process
srun --jobid=<holder> --overlap bash profiling/run.sh --round 0,1,2,3,4,sweep
```

Outputs go to `profiling/results/` (git-ignored): `<round>/<model>.json`,
`<round>/<model>.trace.json`, one `timings.csv` across all runs, and `env.txt`.

## Notes on the engine path

- `Toto2Forecaster` is not recognised as a foundation model by the engine's resolver
  (its keyword list lacks "toto"), so `run_recipe` would treat it as `fit_on_series` and
  run an expanding-window refit backtest. The harness sets `training_regime: zero_shot`
  on Toto's inline spec so all three models take the same single-holdout path; the JSON
  records `engine_default_regime` alongside `regime_used`.
- `Chronos2Forecaster` defaults to bf16 weights; the fp32 rounds pass
  `torch_dtype=float32` explicitly.
- `run_recipe` makes two model calls per request (holdout backtest, then the final
  forecast), so its latency is not comparable one-to-one with a single native forward.
- `Moirai2Forecaster` defaults to `context_length=200`; the harness passes 2048 so it sees
  the same context as the others. Its regime resolves to zero-shot without an override
  ("moirai" is in the resolver's keyword list).

## Notes on Moirai-2.0

- The native path uses sktime's vendored copy (`sktime.libs.uni2ts`), which is what
  `Moirai2Forecaster` runs by default. The `uni2ts` package itself pins `torch<2.5` and
  `numpy~=1.26`, so it is not installed.
- `moirai-2.0-R-small` has patch 16 and `max_seq_len` 512, so context 8192 (512 context
  tokens plus prediction tokens) runs past the trained length. The rotary cache extends
  itself, so the rounds still run, but treat 8192 quality numbers as out of range.
- bf16 (round 3) runs under autocast with fp32 weights, like Toto.
