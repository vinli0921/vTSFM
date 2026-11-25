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
affected origin, preserving NaNs. A missing batch result or wrong result
count uses the native path for the whole chunk. The fitted origin forecasters
remain available for sktime updates.

The Toto callback reuses sktime's cached model and exact context alignment.
Origins batch when their aligned context width, padding and missing-value path,
seed, decode block size and number of forecast patches match. The aligned
targets, masks and series IDs concatenate along the independent batch dimension.
Grouping and the point budget use the input shapes. Each bounded batch allocates
its own padding when it runs. Incompatible or individually oversized origins
run serially. A batch that runs out of memory or trips the linear origin-shape
guard retries those origins with their native serial forecasts.

Batched FP32 projections can round differently from single-origin projections.
A thread-local Torch operation scope runs eligible projections with their
original per-origin matrix shapes. Other operations can share the batch. The
initial checkpoint run preserved the anomaly labels but missed the residual
tolerance. With the projection scope, a follow-up checkpoint run matched the
residual matrix and interval bounds exactly at the original tolerance.
Instrumented forecast API calls were 1,844 serial and 143 batched. The
whole-recipe comparison measures the runtime cost of the resulting execution.

## Capped calibration horizon (#19)

sktime forecasts from every calibration origin to the end of the history. On
the Chiller6 recipe the first origin (204) asks for 1,844 steps. That exceeds
the TimesFM catalog card's 128-point output head. Moirai ran out of memory on a
40 GB A100 in the original profiling round for a different reason: sktime's
Moirai 2 wrapper loads a fresh model copy on every fit, and `ConformalIntervals`
keeps every fitted origin forecaster in `forecasters_` for later updates, so
about 1,844 copies pile up on the GPU.

The intervals read only a short prefix of each origin's forecast. Residual row
`origin` comes from a fit on `y.iloc[:origin]`, so its first column is lead 1.
sktime 1.2.0 reads `np.diagonal(residuals, offset=k)` for relative step `k`,
which is lead `k + 1`. Intervals for `fh = 1..96` therefore read leads 2–97.
Lead 1 and every lead after 97 are computed but never read, by any of
the four interval methods.

`cap_calibration_horizon=True` makes each origin forecast only `max(fh) + 1`
steps, or the shorter remaining suffix near the end. It works with or without
a batch callback.

```python
factory = partial(BatchedConformalIntervals, cap_calibration_horizon=True)
result = run_recipe(store, series, recipe, conformal_factory=factory)
```

The interval residuals are unchanged only if a model's first 97 forecast
values don't depend on the requested length. The opt-in checkpoint test below
checks this per model at `atol=rtol=1e-4`. Residual-matrix entries beyond the
cap stay NaN.

The cap applies only when the horizon is known and positive. These cases keep
sktime's uncapped loop:

- covariates
- origin sampling (`sample_frac`)
- explicit `n_jobs`
- multi-index series

These cases are rejected instead of reading missing residuals:

- an interval request that needs a lead beyond the cap
- `update(update_params=True)` after a capped fit

Refit with the new horizon in either case. Because capped fits reject updates,
they keep no origin forecasters in `forecasters_`, so a wrapper that loads a model
per fit holds one copy at a time per calibration chunk, not one per origin. On
CPU with sktime's Moirai 2 wrapper and 8 origins, 9 model copies stayed alive
after an uncapped fit and 1 after a capped fit.

Per model, through the AssetOps catalog cards:

- **TimesFM 2.5: the test compares 97 against 128 steps, not 1,024.**
  - The checkpoint's `horizon_length` is 128, and it sets the shape of the
    point output head, so this card cannot forecast 1,024 steps. The test checks
    that sktime rejects a 1,024-step request.
  - The sktime wrapper runs one forward pass for any request up to 128 and
    indexes the steps it needs.
  - The 97-step capped request fits the unmodified card. The uncapped recipe's
    1,844-step request never can.
  - The card loads with `device_map`, which needs `accelerate`. The
    `recipes-timesfm2` extra pins `accelerate==1.16.0`; it is separate from
    `recipes` because `accelerate` requires Torch.
- **Moirai 2.0:** the round 4 card (sktime's vendored `Moirai2Forecast`) runs in
  its own environment, `.venv-recipes-moirai2`. Build it with
  `scripts/setup.sh --recipes moirai2` from the `recipes-moirai2` extra.
  - It can't share `.venv-moirai2`. The vendored class returns
    `(preds,), None, None`, and gluonts 0.14.3 (pinned by `uni2ts==2.0.0`)
    calls `.cpu()` on that output. The recipe environment uses gluonts 0.17.0
    and no uni2ts.
  - sktime calls `create_predictor(batch_size)` without a device. gluonts
    resolves `"auto"` to CUDA when it's available, so the predict device follows
    CUDA availability, not `map_location`.
  - The requested length sets the recursive decode length, so prefix equality
    must be measured.
- **Toto 2:** the recipe runs it natively without a horizon limit, and it runs
  only on the team VM with #58's documented recipe install. The cap also
  shortens Toto's calibration forecasts, but it changes the shapes
  `TotoCalibrationBatcher` groups, so the batched timing above does not carry
  over.

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

`tests/test_conformal_horizon_cap.py` checks the cap on CPU with deterministic
stand-in forecasters, with and without a callback:

- intervals match uncapped sktime for every interval method; Range, Datetime
  and Period indexes; and contiguous, gapped and suffix-exceeding horizons
- the read residual diagonals are equal, and each origin's requested length
  equals the cap
- an off-by-one cap (`max(fh)`) changes the intervals
- the fallbacks and rejections listed above

`tests/test_assetops_calibration.py` also runs the pinned recipe in capped and
capped-batched modes.

### Checkpoint checks for the cap

`tests/test_conformal_horizon_models.py` is opt-in and loads one AssetOps
catalog card per run. The harness's `recipe_estimator` cards are used, with the
device configurable. It runs two checks:

- **Prefix match:** at cutoffs 204, 1,000 and 2,048 of the Chiller6 training
  history, the first 97 values of a 97-step forecast must match a 1,024-step
  forecast (128 for TimesFM) at `atol=rtol=1e-4`.
- **Recipe labels:** the capped Chiller6 recipe runs through `run_recipe`. Its
  labels are compared with the uncapped recipe where that run is feasible: always
  for Toto, and for Moirai only with `VTSFM_RECIPE_COMPARE_UNCAPPED=1` (long, may
  run out of memory). TimesFM can't run the uncapped recipe.

The fixture path is chosen in this order:

1. `VTSFM_CHILLER6_JSON`
2. `data/AssetOpsBench/src/couchdb/scenarios_data/shared/iot/chiller_6.json`
3. the same file in the `VTSFM_ASSETOPS_ROOT` checkout

The loader checks both recorded SHA256 values.

Each run writes `results/issue19/<model>.json`, which is git-ignored.
`VTSFM_RECIPE_COMPARE_UNCAPPED=1` writes `<model>-uncapped.json` instead. Each
file records:

- the card
- per-cutoff prefix lengths, maximum absolute and relative differences, and
  pass/fail
- capped and uncapped anomaly indexes where run
- package versions, device and GPU
- git commit, dirty state and a diff hash
- wall times, which include model load

Failures are recorded too. `VTSFM_RECIPE_CUTOFFS` selects a subset of cutoffs,
and `VTSFM_RECIPE_RESULTS` changes the directory. If a model misses the 1e-4
prefix tolerance, keep the cap off for that model rather than loosening it.
Times are recorded, not a speedup claim.

Results on an A100-SXM4-40GB (driver 595.91.07) at `2b298f5`, FP32:

| Model | Prefix max abs / rel difference (3 cutoffs) | Capped recipe | Labels vs uncapped |
| --- | --- | --- | --- |
| TimesFM (97 vs 128) | 0 / 0; 1,024 rejected | 7 anomalies, 1,512 s | uncapped cannot run |
| Toto (97 vs 1,024) | 2.3e-5 / 3.1e-7 | 8 anomalies, 53 s | identical 8 (uncapped 289 s) |
| Moirai (97 vs 1,024) | 2.0e-4 / 3.1e-6 | 8 anomalies, 509 s | not run (optional) |

Every prefix check passed. Before the cap, TimesFM could not run this recipe
and Moirai ran out of memory. Wall times are single runs and include model load.
TimesFM and Moirai calibrate serially, one origin at a time; only Toto has a
batch callback.

### VM runbook for the checkpoint checks

On a fresh Linux VM with an idle CUDA 13 GPU, copy
`issue19-conformal-cap.patch` to the home directory, then run:

```bash
# 1. Code: draft #58's head (ec491d2) plus the #19 patch.
git clone https://github.com/vinli0921/vTSFM.git && cd vTSFM
git fetch origin pull/58/head:pr58 && git checkout pr58
git apply ~/issue19-conformal-cap.patch

# 2. Environments. .venv-toto2 runs Toto and TimesFM, with #58's documented
#    recipe install (it also provides sktime 1.2.0). .venv-recipes-moirai2
#    runs Moirai's catalog card.
./scripts/setup.sh --adapter toto2
./scripts/setup.sh --recipes moirai2
UV=.tools/uv/uv; [ -x "$UV" ] || UV=uv
"$UV" pip install --python .venv-toto2/bin/python -e '.[recipes,recipes-timesfm2,toto2]' \
  'toto-models==1.0.0' 'skpro==2.14.0' 'torch==2.14.0'

# 3. Pinned AssetOpsBench. The tests patch a temporary copy; the checkout
#    also provides the Chiller6 fixture.
git clone https://github.com/IBM/AssetOpsBench.git ../AssetOpsBench
git -C ../AssetOpsBench checkout e844f6468c6c68b328022e7ca8e31da140842cc7
export VTSFM_ASSETOPS_ROOT=$(realpath ../AssetOpsBench)

# 4. CPU recipe tests, then one checkpoint run per model on the idle GPU.
PYTHONPATH=src:tests .venv-toto2/bin/python -m pytest -q tests/test_conformal_horizon_cap.py \
  tests/test_conformal_batching.py tests/test_assetops_patch.py tests/test_assetops_calibration.py
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
VTSFM_RECIPE_MODEL_TEST=timesfm2 PYTHONPATH=src:tests \
  .venv-toto2/bin/python -m pytest -q -s tests/test_conformal_horizon_models.py
VTSFM_RECIPE_MODEL_TEST=toto2 PYTHONPATH=src:tests \
  .venv-toto2/bin/python -m pytest -q -s tests/test_conformal_horizon_models.py
VTSFM_RECIPE_MODEL_TEST=moirai2 PYTHONPATH=src:tests \
  .venv-recipes-moirai2/bin/python -m pytest -q -s tests/test_conformal_horizon_models.py
# Optional, long, may run out of memory: Moirai's uncapped recipe labels.
VTSFM_RECIPE_MODEL_TEST=moirai2 VTSFM_RECIPE_COMPARE_UNCAPPED=1 PYTHONPATH=src:tests \
  .venv-recipes-moirai2/bin/python -m pytest -q -s tests/test_conformal_horizon_models.py -k labels

# 5. Bring these back for the PR.
ls results/issue19/   # timesfm2.json toto2.json moirai2.json [moirai2-uncapped.json]
```

The Toto run includes an uncapped recipe for the label comparison. #58
measured that run at 287 s. TimesFM records capped labels only.

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
