# Toto-2 conformal calibration batching

AssetOpsBench's conformal anomaly recipe uses sktime `ConformalIntervals`.
Calibration fits each growing history prefix and forecasts the remaining
suffix. The optional Toto-2 callback batches compatible point forecasts while
sktime owns the residual matrix, intervals and final full-history forecaster.

## AssetOps integration

The packaged factory patch adds an optional `conformal_factory` keyword through
`run_recipe`, `run_anomaly` and `_conformal_ad`. The default factory is sktime's
`ConformalIntervals`. Apply the packaged patch to a separate AssetOpsBench
checkout at `e844f6468c6c68b328022e7ca8e31da140842cc7`.

```bash
.venv-toto2/bin/python -m vtsfm.recipes.assetops /path/to/AssetOpsBench --check
.venv-toto2/bin/python -m vtsfm.recipes.assetops /path/to/AssetOpsBench --apply
```

The helper works from an installed wheel and needs only Python and Git. Select
the checkout root. Before applying, the helper checks that the tracked
composition file matches the pinned upstream file. Other commits with the same
file are accepted. The default command is read-only. Reapplying the hook is a
no-op, and overlapping changes fail before writing.

Install the recipe dependencies in the Toto Python 3.12 environment alongside
the project's existing dependencies.

```bash
uv pip install --python .venv-toto2/bin/python -e '.[recipes,toto2]' \
  'toto-models==1.0.0' 'skpro==2.14.0' 'torch==2.14.0'
```

Use the checkout's `.tools/uv/uv` if setup bootstrapped uv there. The GPU pair
used Python 3.12.12, Torch 2.14.0+cu130, sktime 1.2.0, skpro 2.14.0,
toto-models 1.0.0, NumPy 2.2.6 and pandas 2.3.3. The tested Toto source pin is
`c079fd7e32424c5d33f9e267c5d32a5630242eb5` for both toto-2 2.0.0 and
dd-unit-scaling 0.1.0. The `toto-models` package supplies the umbrella
distribution expected by sktime.

Pass the batching factory to the upstream recipe.

```python
from functools import partial
from servers.tsfm.engine.composition import run_recipe
from vtsfm.recipes.conformal import BatchedConformalIntervals
from vtsfm.recipes.toto import TotoCalibrationBatcher

factory = partial(BatchedConformalIntervals,
                  batch_predict=TotoCalibrationBatcher(), batch_size=64)
result = run_recipe(store, series, recipe, conformal_factory=factory)
```

The recipe still owns imputation, model-card resolution, horizons, anomaly
decisions and persistence.

## Calibration execution

`BatchedConformalIntervals` retains sktime's constructor arguments and adds
keyword-only `batch_predict=None` and `batch_size=64` (1–256). Its callback
receives fitted forecasters and their future indexes and returns one point
forecast Series per origin. The original sktime loop runs for covariates,
sampled origins, updates, explicit `n_jobs`, or a forecaster the callback cannot
support, including an unseeded Toto forecaster. This preserves native fit and
prediction ordering when RNG state matters.

Malformed and non-finite forecasts use native residual calculation for the
affected origin, preserving NaNs. A wrong result count falls back for the
whole chunk. The fitted origin forecasters remain available for sktime updates.

The Toto callback reuses sktime's cached model and exact context alignment.
Origins batch when their aligned context width, padding and missing-value path,
seed, decode block size and number of forecast patches match. The aligned
targets, masks and series IDs concatenate along the independent batch dimension.
Incompatible or individually oversized origins run serially. A batch that runs
out of memory or trips the linear origin-shape guard retries those origins
with their native serial forecasts.

Batched FP32 projections can round differently from single-origin projections.
A thread-local Torch operation scope runs eligible projections with their
original per-origin matrix shapes. Other operations can share the batch. The
initial checkpoint run preserved the anomaly labels but missed the residual
tolerance. With the projection scope, a follow-up checkpoint run matched the
residual matrix and interval bounds exactly at the original tolerance.
Instrumented forecast API calls were 1,844 serial and 143 batched. The
whole-recipe comparison measures the runtime cost of the resulting execution.

## Checks

Run the recipe tests in an environment with the `recipes` extra. The
AssetOpsBench test uses a pinned source checkout and applies the factory patch
to a temporary copy.

```bash
VTSFM_ASSETOPS_ROOT=/path/to/AssetOpsBench PYTHONPATH=src:tests \
  .venv-toto2/bin/python -m pytest -q tests/test_recipe_imports.py \
    tests/test_conformal_batching.py tests/test_assetops_patch.py tests/test_assetops_calibration.py

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' VTSFM_NATIVE_TOTO_TEST=1 PYTHONPATH=src:tests \
  .venv-toto2/bin/python -m pytest -q tests/test_toto_calibration_native.py tests/test_toto_linear.py
```

The native CPU tests use tiny random-weight Toto models with the real sktime
forecaster. They cover growing contexts, differing horizons, decode blocks
`None`, 4 and 8, native fallbacks and seed order. Residual and interval
comparisons use `atol=rtol=1e-5`.

## Whole-recipe timing

At `6f32c19`, two fresh processes ran the pinned recipe and resolver on an
idle A100-SXM4-40GB (driver 595.91.07), first batched and then native. They
used Toto-2 313M checkpoint revision
`a7bab288f5e95f8606f8306f86659357e1c001ef`, FP32 with TF32 disabled, seed 0,
`decode_block_size=None` and an 18% per-process Torch allocator cap. The
batched run used a calibration batch size of 64.

The original `Chiller 6 Supply Temperature` series was sorted by timestamp,
deduplicated and stripped of missing target observations. Its final 2,144
observed values provided 2,048 training points and 96 holdout points, with
coverage 0.9, empirical intervals and the default initial window of 204.
The source fixture SHA256 is
`8073af23ea71fc401aee97fd40b47678a80ad0549eb4153b1eb0ab8dc9b246e2`.
The selected float64 values hash to
`3d2f4f8d3959eec792b212c457c1dfd5ace7683a76cbdb54754c8af554b1761a`.

| Native | Batched | Speedup | Matching anomaly labels |
| ---: | ---: | ---: | --- |
| 287.0 s | 60.2 s | 4.77× | All 8 |

The timer ran from entry to `run_recipe` through CUDA completion and included
the first model load. Input preparation, artifact writing and forecast-call
instrumentation were outside the measurement. The residual matrices and all
192 interval bounds matched exactly, including missing-value masks, and the
recipe outputs matched apart from the generated run ID. This was one matched
pair on the original Chiller6 fixture.
