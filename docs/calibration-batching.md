# Conformal calibration batching

AssetOpsBench's conformal anomaly recipe uses sktime `ConformalIntervals`. Its
calibration loop fits each growing history prefix and forecasts the remaining
suffix. Both context lengths and horizons vary. This optional implementation
batches compatible point forecasts while retaining sktime's residual matrix,
interval calculations and final full-history forecaster.

## AssetOps integration

The pinned AssetOpsBench source has no interval-estimator injection boundary.
The packaged `vtsfm.recipes/assetops-conformal-factory.patch` adds an optional
`conformal_factory=None` keyword through `run_recipe`, `run_anomaly` and
`_conformal_ad`. Its default selects the original `ConformalIntervals`.

**The external patch is provided here and has not landed in AssetOpsBench.**
Check or apply it to a separate checkout; the tested source commit is
`e844f6468c6c68b328022e7ca8e31da140842cc7`:

```bash
python -m vtsfm.recipes.assetops /path/to/AssetOpsBench --check
python -m vtsfm.recipes.assetops /path/to/AssetOpsBench --apply
```

The helper works from an installed wheel and needs only Python and Git. Omitting
both flags performs the same read-only check. Select the Git checkout root;
subdirectories are rejected to prevent Git from silently skipping the patch.
Before applying, the helper checks the tracked composition source against the
file from the pinned commit. Other commits with that same source file are
accepted.
Already applied hooks are a no-op, unrelated changes are preserved, and
unsupported or overlapping source changes fail before writing. No installed
module is patched, and recipe execution still uses AssetOpsBench's own code.

Install vTSFM's `recipes` extra alongside that project's existing dependencies.
Keep the models in separate environments because they use different versions of
Torch. Both tested environments used sktime 1.2.0, NumPy 2.2.6 and pandas 2.3.3
from the `recipes` extra.

| Callback | Tested model runtime |
| --- | --- |
| Chronos | Python 3.11, Torch 2.9.1+cu128, Transformers 4.57.6, chronos-forecasting 2.2.2 from source `ba47d25a04aedac33af2fb9780609afd54744d5e` |
| Toto | Python 3.12, Torch 2.14.0+cu130, toto-2 2.0.0 and dd-unit-scaling 0.1.0 from source `c079fd7e32424c5d33f9e267c5d32a5630242eb5`, toto-models 1.0.0, skpro 2.14.0 |

Then call the actual AssetOps runner:

```python
from functools import partial
from servers.tsfm.engine.composition import run_recipe
from vtsfm.recipes.conformal import BatchedConformalIntervals
from vtsfm.recipes.chronos import ChronosCalibrationBatcher

factory = partial(BatchedConformalIntervals,
                  batch_predict=ChronosCalibrationBatcher(), batch_size=64)
result = run_recipe(store, series, recipe, conformal_factory=factory)
```

The AssetOps recipe still owns imputation, model-card resolution, horizon policy,
anomaly decisions and persistence. Its existing `empirical` interval method and
all requested forecast horizons are preserved. Calling `run_recipe` with its old
arguments uses the original estimator. There is no copied recipe orchestration
or installed-module monkeypatch.

## Existing sktime behavior

`BatchedConformalIntervals` retains the original constructor argument order and
adds keyword-only `batch_predict=None` and `batch_size=64` (1–256). No callback
uses the original sktime loop. Covariates, sampled origins and updates also use
that loop, as does an explicit `n_jobs` setting. Other interval methods,
initial-window settings, cloning and deferred forecasting horizons remain usable.

The callback receives fitted independent forecaster clones and their original
future indexes, never holdout values. It returns a point-forecast Series per
origin with the same index and order. A malformed or non-finite forecast uses
sktime's native residual calculation for that origin, including any NaNs it
produces. A wrong result count falls back for the whole chunk. Per-origin fitted
forecasters remain available for native sktime updates.

Native callbacks expose `supports_forecaster(forecaster)`. Unsupported models or
models with `seed=None` delegate the **entire** original loop before any origin
is cloned. This preserves the original interleaving of fitting and prediction,
including random-number consumption. A custom callback can provide the same
predicate; otherwise it explicitly opts into fitting a chunk of origins before
forecasting them and must establish that this preserves its model's behavior.

## Chronos execution

The callback groups univariate Chronos contexts with a shared pipeline, explicit
seed, matching forecast settings and matching **native input-patch widths**.
The pinned dataset already left-pads histories and the model pads to a whole
input patch. For non-overlapping input patches, histories in the same bucket
therefore reach the same patched context shape without introducing an extra
token. Other model configurations retain exact-width groups.

Groups also require equal **complete output-patch counts**. The pinned pipeline
computes whole patches and crops afterward, so these origins share the same
forecast operation even when their requested horizons differ within a patch.
For a model with 16-point output patches, horizons 61–64 each compute four
patches; horizons across a patch boundary remain separate. Long horizons retain
the complete native unrolling schedule, including its nine quantile paths, and
their expanded work counts toward the point budget.

Growing prefixes can batch without changing their configured context length.
The callback passes their prepared contexts unchanged to native collation.
Cross-learning, covariates and other models use serial forecasts. The point
budget splits optimized calls; individually oversized origins also run serially.
There is no general horizon-prefix or arbitrary ragged-padding assertion flag.

This implementation provides actual native batch calls for compatible origins.
Its benefit depends on the existing context window, fixed seed and horizon mix;
workloads without compatible origins remain serial. The real-checkpoint CUDA
record below includes the initial residual mismatch and its numerical fix.
Both follow-up runs also pass the unchanged residual-scale tolerance.
See [whole-recipe timing](#whole-recipe-timing) for the matched latency comparison.

## Toto execution

For Toto-2 recipes, use `TotoCalibrationBatcher()` from `vtsfm.recipes.toto` in
the same factory. Use Python 3.12 and install the real upstream dependencies:

```bash
uv pip install --python .venv-toto2/bin/python -e '.[recipes,toto2]' \
  'toto-models==1.0.0' 'skpro==2.14.0' 'torch==2.14.0'
```

Use the checkout's `.tools/uv/uv` if setup bootstrapped uv there. The explicit
interpreter keeps this installation inside the existing checkout environment.

[`toto-models` 1.0.0](https://pypi.org/project/toto-models/1.0.0/) is Datadog's
official umbrella requiring `toto-2>=2.0.0`; it is compatible with this project's
pinned `toto-2` 2.0.0 source at `c079fd7e32424c5d33f9e267c5d32a5630242eb5`.
Installing the genuine umbrella and `skpro` satisfies sktime's ordinary checks;
the callback and native tests keep those checks enabled. The `recipes` extra
alone remains a lightweight calibration dependency set.

The callback reuses sktime's cached model and exact `_align_context` outputs.
Origins batch only when their aligned context width, padding/missing-value path,
seed, decode-block setting and total output-patch count match. Their prepared
targets, masks and series IDs concatenate along the independent batch dimension.
No additional padding is introduced. The native median and original indexes are
retained; incompatible origins and individually oversized work run serially.

The native CPU suite checks growing contexts, varying horizons and decode-block
sizes `None`, 4 and 8 using a tiny random-weight official Toto model. Compatible
origins share forecast calls while retaining the original residuals and intervals.
The real-checkpoint CUDA record and timings below cover the original anomaly
recipe.

## Numerical execution

FP32 projections can round differently when independent origins become one
larger matrix. For these callbacks, linear projections retain each origin's
native matrix shape; Chronos also retains its original normalization width and
buffer alignment. Long horizons retain each origin's quantile paths together.
A thread-local Torch operation scope confines this behavior to the batch call
without replacing methods on sktime's shared cached models or changing global
backend settings. These boundaries follow the pinned upstream model layouts.

Other operations still run in batches. Fewer forecast API calls do not mean all
linear kernels have been combined: per-origin projections and Python dispatch
may limit the benefit. The whole-recipe timing below measures the resulting cost.

## Checks

```bash
PYTHONPATH=src:tests python -m pytest tests/test_recipe_imports.py \
  tests/test_conformal_batching.py tests/test_chronos_calibration.py -q

VTSFM_ASSETOPS_ROOT=/path/to/AssetOpsBench PYTHONPATH=src:tests \
  python -m pytest tests/test_assetops_calibration.py -q

CUDA_VISIBLE_DEVICES='' VTSFM_NATIVE_CHRONOS_TEST=1 PYTHONPATH=src:tests \
  python -m pytest tests/test_chronos_calibration_native.py -q

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' VTSFM_NATIVE_TOTO_TEST=1 PYTHONPATH=src:tests \
  python -m pytest tests/test_toto_calibration_native.py -q

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' VTSFM_NATIVE_CHRONOS_TEST=1 \
  VTSFM_CHRONOS_ORIGINAL_SHAPE_TEST=1 PYTHONPATH=src:tests \
  python -m pytest tests/test_chronos_calibration_native.py -q
```

Recipe tests compare actual original upstream code with a temporary source copy
containing only the factory patch. They cover default, explicitly serial and
batched estimators, labels, metadata, persistence, horizons and imputation. The
optional native test uses the actual sktime forecaster and Chronos pipeline with
a tiny CPU model and random weights; it checks residual/interval parity and fewer
native calls under eager and SDPA attention. It downloads no checkpoints and is
not a performance measurement. The Toto fixture supplies tiny random weights
through the native cache and requires the ordinary dependencies, with no tag
overrides. Normal lightweight CPU CI skips optional model and sktime tests;
importing `vtsfm.recipes` needs none of those packages.

## Recorded native CPU evidence

The original-shape check uses all growing prefixes of a synthetic 2048-point
training series, matching round 4's 2144 observations minus its 96-point holdout.
It retains `method='empirical'`, coverage 0.9, sktime's default 204-point initial
window, forecaster seed 0, native context 8192, 16-point input/output patches and
a 1024-point native horizon. No context or horizon cap is introduced. Random
tiny-model weights use Torch seed 42, nine quantiles, one SDPA layer with width
8, CPU FP32 operations, and no downloaded checkpoint. At batch size 64 and point budget
1,048,576, native forecast calls fell from **1844 to 324**. Every residual and
the 90% intervals matched serial execution at `atol=rtol=1e-5`, including horizons
above 1024. This records call counts and correctness, without a latency claim.

That check used sktime 1.2.0, NumPy 2.2.6, pandas 2.3.3, Torch 2.9.1+cu128 on
CPU, and Chronos source `ba47d25a04aedac33af2fb9780609afd54744d5e` (installed
distribution 2.2.2). The native pipeline/model files match that pinned checkout.
Smaller growing-prefix cases cover capped and uncapped contexts under both eager
and SDPA attention, using the same parity tolerances.

Toto's native checks passed in a standalone Python 3.12.12 environment with
Torch 2.14.0+cpu, torchvision 0.29.2+cpu, toto-2 2.0.0 and dd-unit-scaling 0.1.0
built from source `c079fd7e32424c5d33f9e267c5d32a5630242eb5`, the real
toto-models 1.0.0 umbrella, skpro 2.14.0, sktime 1.2.0, NumPy 2.2.6 and pandas
2.3.3. A normal resolver installation and `uv pip check` passed without borrowed
model-environment paths, fake distributions or dependency-tag overrides. Tiny
random-weight native tests cover decode-block settings `None`, 4 and 8,
bounded partial batches, serial fallbacks and the original seed-before-load
ordering. Residual and interval comparisons use `atol=rtol=1e-5`.

The CUDA normalization regression can also run without checkpoint downloads:

```bash
python -m pytest tests/test_chronos_ops.py -m gpu -q
```

## Initial real-checkpoint CUDA checks

On November 10, 2026, separate stock-sktime and batched processes ran the actual
pinned AssetOpsBench recipe and resolver through the optional factory patch.
The original Chiller6 target was `Chiller 6 Supply Temperature`, sorted by
timestamp, deduplicated and stripped of missing target observations. The final
2144 observed values supplied 2048 training points and a 96-point holdout, with
an unnamed RangeIndex Series, no imputation/covariates, seed 0, coverage 0.9,
the empirical interval method and default initial window 204. Contexts and
calibration horizons were unchanged. The source fixture SHA256 is
`8073af23ea71fc401aee97fd40b47678a80ad0549eb4153b1eb0ab8dc9b246e2`;
the selected float64 values hash to
`3d2f4f8d3959eec792b212c457c1dfd5ace7683a76cbdb54754c8af554b1761a`.

Before the arithmetic scopes above were added, both real checkpoints retained
every anomaly label and all recipe outputs
except the generated run ID. Point forecasts reconstructed from the saved
residuals and observed targets, and all 192 interval bounds, passed the unchanged
`atol=rtol=1e-5` check. A separate comparison applying that same relative tolerance
to the residual values themselves **failed**; its raw failures remain recorded.

| Model | Matching anomaly labels | Forecast / interval failed values | Strict residual failed values | Maximum absolute difference |
| --- | --- | --- | --- | --- |
| Chronos-2 | All 11 original anomalies | 0 / 0 | 266 of 1,701,090 finite residuals | 0.000313 |
| Toto-2 313M | All 8 original anomalies | 0 / 0 | 7117 of 1,701,090 finite residuals | 0.000145 |

Subtracting the observed value changes the scale used by a relative-error test.
The largest forecast error divided by `1e-5 + 1e-5 * abs(native_forecast)` was
0.855 for Chronos and 0.203 for Toto, while the corresponding residual-scale
ratios reached 3.641 and 7.073. This explains why the gates give different
results; it does not establish the kernel-level cause of the differences.
No tolerance or production numerical setting was changed to obtain a pass.

Chronos used checkpoint `29ec3766d36d6f73f0696f85560a422f50e8498c`, the
Chronos/sktime pins above, Torch 2.9.1+cu128 and Transformers 4.57.6, with native
SDPA FP32, TF32 disabled and context length 8192. The real checkpoint has **21
quantiles**, unlike the nine-quantile tiny CPU model. Instrumented calibration
pipeline calls were **1844 to 915**, with batches of 1, 2, 4, 11 or 15 origins
under the unchanged 1,048,576-point budget. All singleton origins were bitwise
equal; the stricter residual failures arose in multi-origin calls.

Toto used checkpoint `a7bab288f5e95f8606f8306f86659357e1c001ef`, source
`c079fd7e32424c5d33f9e267c5d32a5630242eb5`, Torch 2.14.0+cu130 and the normal
Python 3.12.12 recipe environment (sktime 1.2.0, skpro 2.14.0, toto-models 1.0.0,
NumPy 2.2.6 and pandas 2.3.3), with FP32, TF32 disabled and
`decode_block_size=None`.
Its native model-call counts were **not instrumented and remain unmeasured**;
they are not inferred from planner metadata. Both interval estimators retained
1844 fitted calibration forecasters.

These untimed correctness checks ran on an A100-SXM4-40GB (driver 595.91.07),
alongside authorized existing workloads, with an 18% per-process Torch allocator
cap. Raw residuals, intervals, labels, provenance, occupancy and process-cleanup
records were retained in ignored experiment results. They establish this
fixture's anomaly identity and forecast/interval checks, not general bitwise
FP32 batching invariance or the issue's under-10-second target. The external
AssetOps patch and manual latency validation remain open; this is related to
issue #18 rather than a full closure claim.

## Follow-up native-arithmetic checks

The revised callbacks were checked at `35ab80a` using fresh stock-sktime and
batched processes on the same fixture, checkpoint pins, runtime settings and
unchanged `atol=rtol=1e-5` gate above. These are native recipe comparisons; they
do not exercise the vLLM HTTP serving path.

Chronos now has **zero failing residuals** among all 1,701,090 finite values,
with maximum absolute difference 0.00001145 and identical missing-value masks.
All 192 interval bounds are bitwise equal. All 11 original anomaly labels and
all recipe outputs except the generated run ID match exactly. Instrumented
calibration pipeline calls remain 1844 serial versus 915 batched; per-origin
projection operations remain separate inside those batch calls.

Toto's **entire residual matrix and all 192 interval bounds are bitwise equal**,
with zero failing values and identical missing-value masks. All 8 original
anomaly labels and the recipe outputs except run ID match exactly. Actual
calibration model calls were instrumented in both fresh processes: **1844 serial
versus 143 batched**, with batch sizes 1, 11, 20 and 31. These are forecast API
calls, not combined linear-kernel counts.

The native controls, model settings, context/horizon schedules and tolerances
were unchanged. Original failures remain recorded separately. A single-operation
`bmm`/`baddbmm` projection alternative was also tested and rejected after six
median residuals in a representative 31-origin Toto batch failed the same gate.
It is not part of the callbacks.

A small checkpoint-free CUDA regression checks Chronos reductions on growing,
unaligned histories. Native CPU tests also cover the scoped projection axes,
long-horizon context cap, cached-model state, thread isolation and exception
cleanup. The installed wheel supplies the external factory patch and its
explicit checkout helper; that patch still has not landed upstream.

These checks resolve the recorded residual-scale failures on the original
fixture. The matched timing below uses the same numerical gate.

## Whole-recipe timing

At `6f32c19`, the original Chiller6 recipe ran on an idle A100-SXM4-40GB with
the checkpoints, fixture, FP32 settings and residual tolerance above. Each
mode used a fresh process. The wall-time interval started at entry to
`run_recipe` and ended at CUDA completion, including the first model load.
Fixture preparation and writing result artifacts were outside that interval.
Calibration calls were timed without the call-count instrumentation.

| Model | Native | Batched | Speedup | Matching anomaly labels |
| --- | ---: | ---: | ---: | --- |
| Chronos-2 | 121.0 s | 81.3 s | 1.49× | All 11 |
| Toto-2 | 287.0 s | 60.2 s | 4.77× | All 8 |

There was one completed pair per model, in native-then-batched order for
Chronos and batched-then-native order for Toto. A Chronos attempt interrupted
by foreign GPU activity was excluded. All retained runs used the same model
settings within each pair, batch size 64 for the batched mode, and the same
18% Torch allocator cap. A fresh comparison of the residuals passed
`rtol=atol=1e-5`, with matching missing-value masks. All interval bounds
matched exactly.

To reproduce, time the stock recipe and the batched factory on the same
checkpoint and fixture in separate processes, synchronizing CUDA immediately
before and after each `run_recipe` call. These timings cover the Chiller6
recipe only. Chronos is still above issue #18's under-10-second target.
