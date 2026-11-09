# Batched conformal calibration (draft)

Issue #18 targets AssetOpsBench's conformal anomaly recipe. At the pinned
AssetOpsBench revision `e844f6468c6c68b328022e7ca8e31da140842cc7`,
`composition._conformal_ad` uses sktime 1.2.0 `ConformalIntervals`. Its expensive
calibration loop fits a clone at each growing history prefix and forecasts the
remaining suffix. The contexts **and** requested horizons therefore change at
every origin. Batched forecasting must preserve both.

This opt-in recipe entry point replaces that loop while leaving sktime's signed
residuals, interval quantiles, coverage, full-history forecast, and AssetOpsBench's
anomaly labels and result records unchanged. In particular, AssetOpsBench uses
the default `empirical` interval method even when the recipe says `conformal`.
The existing requested horizons are neither capped nor chunked (#19).

## Use

Install vTSFM's `recipes` extra in the separate AssetOpsBench environment. The
integration lazily imports AssetOpsBench; it does not patch either installation.
The pinned conformal function's source hash is checked before use; upstream
recipe changes require revalidating this boundary.

```python
from vtsfm.recipes.assetops import run_recipe
from vtsfm.recipes.chronos import ChronosCalibrationBatcher

batch_predict = ChronosCalibrationBatcher()
result = run_recipe(store, series, recipe, batch_predict=batch_predict, batch_size=64)
```

The wrapper has the normal AssetOpsBench `run_recipe` arguments plus an explicit
`batch_predict` callback and a calibration batch limit (1–256, default 64). Other
recipe tasks delegate to the original entry point. The batched conformal path
requires a zero-shot, univariate forecaster with no exogenous inputs, origin
sampling, or updates. Invalid batch outputs fail instead of corrupting intervals.

`BatchedConformalIntervals` passes fitted independent clones and their original
future indexes to `batch_predict(forecasters, future_indexes)`. The callback returns one
finite point-forecast Series per origin, preserving the target index and order.
It can perform actual model batching without changing sktime's calibration math.

## Chronos bridge and remaining qualification

The included bridge reuses each fitted sktime `Chronos2Forecaster`'s already-loaded
pipeline, exact prepared context, seed and forecast settings. It selects the
native median, disables cross-learning, and enforces an input/output-point budget.

**Safe defaults keep different context widths and horizons in separate calls.**
For a single growing-prefix calibration series these defaults remain sequential.
To batch its origins, the model must first demonstrate both ragged-context
equivalence and horizon-prefix invariance against isolated forecasts. Only then
may callers explicitly set `context_padding_invariant=True` and
`horizon_prefix_invariant=True`. The API never assumes that requesting a longer
forecast and cropping it yields the same predictions. Both declarations must be
requalified after changing model/runtime settings. The candidate's padding flags
are not enabled by the serving CLI or normal recipe path.

This draft establishes the bounded residual-loop integration and native callback
boundary. Real Chronos checkpoint parity, the original anomaly-count fixtures,
Toto's model-specific batch callback, and the requested GPU latency improvement
are **pending**. No all-model or serving speedup is claimed. CPU tests use a
deterministic forecaster and a stand-in pipeline; they do not qualify model flags.

## Checks

```bash
PYTHONPATH=src:tests python -m pytest tests/test_recipe_imports.py \
  tests/test_conformal_batching.py tests/test_chronos_calibration.py -q

VTSFM_ASSETOPS_ROOT=/path/to/AssetOpsBench PYTHONPATH=src:tests \
  python -m pytest tests/test_assetops_calibration.py -q
```

The second suite executes the actual pinned AssetOpsBench recipe source with
only estimator resolution replaced by a deterministic forecaster. It compares
labels, metadata, persistence, default/explicit horizons and imputation against
the original recipe. Use an environment with the `recipes` extra; normal CPU CI
skips optional sktime tests. `vtsfm.recipes` and the wrapper remain importable
without NumPy, pandas, sktime, Torch, vLLM, or AssetOpsBench.
