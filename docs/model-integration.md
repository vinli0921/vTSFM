# Manual GPU model validation

Run this suite on a Linux NVIDIA GPU host before merging a model or serving
change. It compares the pinned TimesFM 2.5, Toto 2, and Moirai 2.0 native GPU
forecasting paths with the vTSFM HTTP path. Three profiles share model lifecycle,
workloads, revision metadata, resource refusal, and cleanup:

| Profile | Purpose | Typical trigger |
| --- | --- | --- |
| `integration` (default) | Native/vLLM numerical and HTTP correctness | Serving/model PRs |
| `accuracy` | Observed-label MAE, MASE, SQL and seasonal-naive comparison | Model/precision/approximation changes |
| `performance` | Matched HTTP throughput, p50/p95, and sampled GPU memory | Optimization experiments |

The runner is manual. It does not start a cloud VM, connect over SSH, install
drivers, or configure a GitHub Actions runner. CPU Actions checks continue to
run separately. Optional Modal contract checks, when available, exercise a
different boundary and do not replace these model/checkpoint checks.

## Prepare the selected model environments

Use a checkout of the commit you intend to test and an idle GPU with a free
loopback port. The driver must already support CUDA. Install each model's pinned
dependencies with the existing setup script:

```bash
./scripts/setup.sh --adapter timesfm25
./scripts/setup.sh --adapter toto2
./scripts/setup.sh --adapter moirai2
```

These create `.venv-timesfm25`, `.venv-toto2`, and `.venv-moirai2`. The integration
runner uses each selected environment's Python; it does not require the CPU
`.venv`. Model checkpoints may download during the first test and need network
access, disk space, and time. Keep caches and weights outside Git.

## Run all models or one adapter

```bash
./scripts/test-models.sh --gpu 0 --port 18001
```

By default the runner checks `timesfm25`, `toto2`, and `moirai2` sequentially. It
validates that all selected environments exist before launching any tests. A
missing dependency or unavailable GPU is a failure, not a skipped success.

For one model or an explicit output directory:

```bash
./scripts/test-models.sh --adapter toto2 --gpu 0 --port 18001 \
  --output results/model-integration/toto-before-change
```

Use `--forecast-execution compile` to exercise the experimental neural compiler
in any profile. Native references and the native HTTP backend stay eager; saved
evidence records the per-backend mode. Compile integration also adds the
retained-shape/full-horizon regression described in
[compiled execution](compiled-execution.md), along with the recorded current-stack
correctness checks. Performance comparisons remain manual.

`--gpu` is a numeric physical GPU index, used consistently for CUDA visibility
and GPU availability checks. The runner sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
FP32/TF32 controls, and four CPU threads. Use a different free port if needed.
It never stops an existing server or another person's GPU workload to make a
test pass. Cancellation should release only test-owned processes; inspect the
saved logs if a run is interrupted.

The default output is a new
`results/model-integration/<UTC timestamp>-<short commit>/` directory. Existing
output directories are refused so previous evidence is preserved. Relative
`--output` paths resolve from the directory where you invoked the runner;
absolute paths and paths containing spaces are supported. The default `results/`
tree is ignored by Git; keep custom artifacts out of commits as well.

## What the test exercises

For each adapter, its native reference runs in a separate subprocess on the
selected GPU, then releases GPU resources before the vLLM server starts. The
test sends numerical requests to the actual HTTP endpoint and compares returned
forecasts with the reference using matching inputs, checkpoint revision, CUDA
device, FP32 settings, horizon, quantiles, and model options. Model-specific
normalization, padding, decoding, and quantile semantics remain part of the
reference behavior.

The integration profile sends six requests per model, in order:

1. One 128-point history, a 17-point forecast, and quantiles 0.1/0.5/0.9.
2. Two unequal histories of 97 and 163 points, a 33-point forecast, and only
   quantiles 0.2/0.8, checking padding, item IDs, and channel selection.
3. A longer forecast: 256 points for TimesFM, 800 for Toto, and 160 for Moirai,
   with all nine quantiles. Toto's case exceeds its 768-point decoding block.
4. The first request repeated after the horizon and batch shape changes.
5. Two additional pinned AssetOpsBench cases, each with two observed sensor
   histories of 128 points and 32 fully observed future labels. All nine quantiles
   are checked. Sensor windows share one compressor; they are not independent assets.

The first profile run prepares the pinned MetroPT fixture in `data/` when absent,
using the existing preparer and its source checksum. Use `--data-dir DIRECTORY`
for a different prepared cache. Existing corrupt or inconsistent data is rejected,
not silently replaced. Input histories may contain the documented causal forward
fill; held-out accuracy labels exclude those imputed rows.

The initial numerical tolerance is `rtol=1e-4`, `atol=1e-4`. Treat that as a
provisional threshold until the tests are validated on the target GPU; do not
silently loosen it to obtain a pass. A native-reference comparison establishes
parity for the tested settings. It does not independently establish forecast
accuracy, safe new batching semantics, or throughput/latency improvements.

The GPU fixture checks availability and owns cleanup of its reference and server
processes. Ordinary adapter failures are logged and the runner attempts the
remaining selected adapters; its final exit status is nonzero if any adapter or
log write failed. An interrupt stops the run rather than starting another model.
The wrapper forwards cancellation to its own pytest process so the fixture can
clean up its GPU children, allows up to 90 seconds for that cleanup, and drains
its logger. If cleanup exceeds the deadline, it terminates the owned test/logger
processes and reports a failure; inspect the cleanup records before using the GPU
again. It never signals unrelated workloads.

The underlying command for each environment is:

```bash
VTSFM_TEST_ADAPTER=toto2 .venv-toto2/bin/python -m pytest -m gpu \
  tests/test_model_integration.py -q -s --junitxml /new/output/junit.xml
```

Prefer `scripts/test-models.sh` for normal use: it also sets the physical GPU
index, port, output path, precision controls, and checkout import paths expected
by the fixture.

## Review the saved evidence

Each adapter directory contains:

- `pytest.log` and `junit.xml` for the test outcome.
- `run.json` for provenance, comparison error statistics, final status, and the
  GPU availability check after cleanup.
- `native.json` / `native.log` and `vllm.log` for reference and server diagnostics.
- Numbered request/response artifacts and owned-process/cleanup records.

The top-level `summary.tsv` records the adapter, pytest exit code, and log-write
exit code. A printed summary alone is not evidence of a pass: inspect all exit
codes and numerical comparisons.

The JSON evidence records the full source commit, dirty working-tree state,
dependency/checkpoint versions, GPU/runtime settings, and comparison results.
Run from a clean, committed checkout for reviewable pre-merge evidence. If testing
uncommitted edits, retain that fact and the diff with the results. Link the
artifacts and source commit in the implementation PR; rerun after relevant code,
pin, precision, or configuration changes.

This documents a manual pre-merge practice. It is not an automatically enforced
branch-protection gate, and CPU test success does not substitute for a completed
GPU model run.

## Accuracy profile

The built-in accuracy slice evaluates five chronological AssetOps cutoffs with
all seven channels. It compares native, vTSFM, and seasonal-naive forecasts:

```bash
./scripts/test-models.sh --profile accuracy --gpu 0
```

MAE and MASE use the 0.5 quantile as the point forecast, not an invented mean.
SQL averages pinball losses over quantiles 0.1 through 0.9 and scales using
seasonal differences from the input history only. Missing future observations
are masked. Zero or insufficient seasonal scales produce explicit null scaled
scores and counts; they do not become successful zero-error observations.
`accuracy.json` retains per-series results and served-minus-native differences.
The seasonal-naive comparison repeats the previous season deterministically at
every quantile; it is not the probabilistic baseline from a published leaderboard.
Current exact paths require native numerical parity; there is no requirement
that a foundation model beat seasonal naive on every small case. Approximate
execution needs a separately justified quality budget, not silently relaxed tests.

For a broader custom smoke slice, prepare pinned data with optional
[`fev`](https://github.com/autogluon/fev) 0.10.0 in the CPU environment:

```bash
./scripts/setup.sh --cpu --evaluation
.venv/bin/python -m benchmarks.prepare_data --output data
.venv/bin/python -m benchmarks.model_evaluation prepare --data-dir data \
  --fev --output data/accuracy-fev.json
./scripts/test-models.sh --profile accuracy --cases data/accuracy-fev.json --gpu 0
```

The optional manifest adds M4 hourly, electricity hourly, and solar hourly:
two rolling windows with up to eight finite univariate series per task. The
dataset revision and configuration are pinned in `benchmarks/fev_smoke.json`;
the manifest freezes the selected IDs, cutoffs, context limits, targets, and
seasonality. No covariates or jointly modelled multivariate tasks are implied.
This is a documented subset for regression checks, not a full fev-bench or
GIFT-Eval score. Keep fev preparation in the CPU environment so it does not
change the three model environments' dependency baselines.

### AssetOps signals

`benchmarks/assetops_sources.json` pins every AssetOpsBench sensor file used
here (commit and SHA256) and names the signals built from them. The default
signals are Chiller 6 supply temperature and condenser water flow, MetroPT oil
temperature and motor current, and hydraulic `PS1` pressure. `prepare_data`
prepares MetroPT by default. `--assetops-signals` also fetches the files used
by the named signals. It verifies and reuses any requested files already in
`data/source/`. Add `--assetops-checkout PATH` to copy missing files from a
local AssetOpsBench checkout. Use `--source-file PATH` to prepare only MetroPT
offline from a verified local copy. To evaluate another `iot/*.json` signal,
pin its file and add a signal entry.

```bash
.venv/bin/python -m benchmarks.prepare_data --output data --assetops-signals
.venv/bin/python -m benchmarks.model_evaluation prepare --data-dir data \
  --assetops-signals --output data/accuracy-signals.json
./scripts/test-models.sh --profile accuracy --cases data/accuracy-signals.json --gpu 0
```

Each window is a C512/H96 forecast (by default) whose whole horizon was
recorded. By default (`--signal-windows non-overlapping`) a signal contributes
every such window whose forecast horizon does not overlap another, chosen
greedily from the earliest eligible origin; histories may overlap. At C512/H96
that gives 24 windows each for Chiller 6 supply temperature and condenser
flow, 11 each for MetroPT oil temperature and motor current (gaps leave only
503 eligible origins), and 17 for hydraulic PS1. The 20+ windows per signal
asked for in [#84](https://github.com/vinli0921/vTSFM/issues/84) are therefore
not attainable for MetroPT and hydraulic at this context and horizon; tables
report each signal's window count instead. `--signal-windows 5` reproduces the
November 14 baseline's five evenly spaced origins. A request carries at most
`--signal-rows-per-case` windows (default 8, Moirai's per-request row limit;
TimesFM and Toto allow 16), so larger signals become several
`...-part-<n>` cases. Item IDs stay `<signal>/origin-<slot>` and tables group
all parts of a signal. Readings are placed on the signal's time
grid (15 minutes for Chiller 6, hourly for MetroPT) or one slot per hydraulic
cycle. Missing history slots are causally forward-filled. The case provenance
records their positions and counts for each window. Every scored future slot
is observed. Seasonality is 96, 24 and 1 respectively.
Hydraulic cycles have no calendar meaning, so its seasonal naive is last-value
naive and its MASE scale is the one-cycle change; compare models with each
other and with naive there, not with 1.

An accuracy run writes `accuracy_table.md` and `accuracy_table.csv` after the
selected adapters finish, plus `accuracy_table_windows.csv` (one row per model
and window: origin, flags, and MAE/MASE/SQL for native, vTSFM, seasonal naive
and mean of history) and `accuracy_table_comparison.csv`. The tables summarize
all available scores from the run, including native MAE, MASE and SQL per model
and signal, seasonal-naive and mean-of-history scores, counts of scaled,
zero-scale and off-state windows, and the largest served-minus-native SQL
difference, plus the share of each signal's history that was forward-filled.
Combining runs at different precisions produces a model × signal × dtype table
with model spread and reduced-precision deltas.

The mean-of-history baseline repeats the mean of the recorded (not
forward-filled) history values at every step and quantile, next to seasonal
naive. Runs made before it existed get it recomputed from their `cases.json`.

`MASE`/`SQL` remain the equal-weight mean of per-window ratios, which one
window with a tiny seasonal scale can dominate (hydraulic PS1 scales span
0.026–2.49 bar). Two aggregates resist that: the ratio of means (summed error
over summed scale) and the geometric mean of per-window MASE. Undefined scales
stay explicit with counts in every aggregate. The paired comparison ranks the
three models and both baselines within each window by median MAE, then gives
95% percentile intervals from a fixed-seed paired bootstrap over windows
(`--resamples`, default 2000; `--seed`, default 0): each resample draws windows
with replacement and scores every competitor on the same draw. A competitor is
"higher than lowest" only when its interval for the ratio to the lowest ratio of
means lies wholly above 1; otherwise it is "not distinguishable". Windows from
one asset are correlated, so these intervals are approximate, and signals with
fewer than five windows get no claim.

A window is flagged off-state when at least half of its observed future labels
are exactly zero (an asset switched off, e.g. condenser flow); the comparison is
repeated without those windows. Chiller 6 supply temperature is noise-like
(lag-1 autocorrelation −0.01 to 0.12; the history mean scores within the models'
range), so it is shown as unforecastable and excluded from ranking claims.
`--scale observed` re-expresses MASE and SQL over seasonal pairs whose two
history values were both recorded, so forward-filled runs cannot shrink the
scale. It rescales the saved scores exactly from the run's `cases.json`; no
rerun is needed. `score_forecasts(..., scale="observed")` applies the same rule
when scoring directly.

```bash
python3 -m benchmarks.accuracy_table results/model-accuracy/FP32_RUN \
  results/model-accuracy/BF16_RUN --output results/accuracy-signals
```

SQL is the harness's quantile approximation of a scaled CRPS. Windows come from
one asset per signal and are correlated; treat the table as evidence about
these signals, not a model ranking. Make per-signal ranking claims only from the
paired comparison. Reduced-precision runs for
[#23](https://github.com/vinli0921/vTSFM/issues/23) should use the same
manifest and table generator as the FP32 run they are compared with.

An accuracy run scores every usable case before exiting with a nonzero status
for any parity failure. The table marks affected signals. On the reference A100
with non-overlapping windows, all three adapters fail the `rtol=atol=1e-4` gate
on 3 of 13 cases, all on condenser water flow: forecasts near zero in a series
reaching about 10,000 differ from native by a few FP32 rounding steps. The
CPU-built rotary-table fix ([#90](https://github.com/vinli0921/vTSFM/pull/90))
addresses Toto and Moirai only; TimesFM's failure needs the magnitude-aware floor
in [#71](https://github.com/vinli0921/vTSFM/issues/71). The table reports native
scores, so the failures do not change the accuracy numbers. See the
[November 23 baseline](baselines/2026-11-23-assetops-signal-accuracy-nonoverlap/README.md).

## Performance profile

```bash
./scripts/test-models.sh --profile performance --gpu 0 \
  --batches 1,4 --contexts 128,512 --horizon 96 --clients 1,4 \
  --requests 100 --warmups 5
```

Each model runs a native HTTP service and vLLM sequentially on the same GPU and
loopback port. Both receive identical pre-encoded numerical payloads and return
the requested quantiles and item IDs. Native execution calls the official model
directly, formats its arrays without an artificial JSON round trip, and permits
one active forecast, matching the current vLLM `max_num_seqs=1` baseline.
Concurrency means outstanding HTTP clients; it does not imply live cross-request
batching. Numerical checks, including a concurrent parity burst, happen outside
the timed measurement. Startup and model downloads are excluded; warmups finish
before measurement and NVML sampling.

`performance.json` and individual `points/` files retain latencies, failures,
request/series throughput, and memory samples. Throughput is successful completions
divided by total wall time under concurrency. A failed request invalidates a
point's throughput; queued requests stop, and interrupted runs retain partial
samples. Memory is NVML whole-device residency sampled every 20 ms, which can
miss short peaks; it is not Torch allocated/reserved memory.

Fewer than 100 measured requests is labelled `smoke_only`; it verifies the
harness rather than a paper result. Repeat independent runs on an idle GPU and
counterbalance `--backend-order native,vtsfm` with `--backend-order vtsfm,native`
before interpreting speed differences. No performance threshold blocks merging
automatically. The older `benchmarks.run` Chronos comparator remains unchanged;
this profile supplies the matched HTTP path for these three adapters.
