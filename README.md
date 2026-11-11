# vTSFM

A general time-series foundation model (TSFM) extension for vLLM, with
Chronos-2 plus minimal TimesFM 2.5, Toto 2 313M, and Moirai 2.0 small adapters.
CPU-safe adapter metadata selects each model family's implementation, numerical
IO, and serving settings.
The `vtsfm` plugin uses the pinned Chronos-2 integration; the original `chronos2`
plugin remains available as a comparator.

Shared batching uses a CPU-safe planner for compatibility, request and row
budgets, and output routing. Numerical adapters can opt into executing compatible scheduled
requests together. Model-specific normalization, patching, covariates, group
semantics, and attention kernels belong to each adapter. The
[historical results](docs/baselines/2026-10-20/README.md) measure the community
Chronos plugin against native inference, before vTSFM-specific optimization.

## Get started on a CPU

Clone the repository using an account with access to this private project.

```bash
git clone https://github.com/vinli0921/vTSFM.git
cd vTSFM
./scripts/setup.sh --cpu
./scripts/doctor.sh --cpu
.venv/bin/python -m pytest -m 'not gpu'
```

Setup creates a `.venv` inside this checkout. It uses an existing `uv` or
bootstraps a versioned copy into `.tools/`; it does not change shell profiles.
CPU mode supports tests, data preparation, and report generation. It does not
install Torch/vLLM or run model inference. Python CLI help is available without
loading CUDA:

```bash
.venv/bin/python -m vtsfm.cli --help
.venv/bin/python -m vtsfm.cli adapters
.venv/bin/python -m vtsfm.cli serve --adapter chronos2 --dry-run
.venv/bin/python -m benchmarks.run --help
```

`adapters` prints JSON metadata without loading Torch, vLLM, or model packages.

Optional [Modal GPU checks](docs/gpu-testing.md) run CUDA contract tests on relevant
same-repository pull requests and on demand.
Set up the documented credit/spend limits and repository secrets before enabling
the workflow. Full-model validation and benchmarks remain manual.

## Serve on a GPU

Use a Linux machine with an NVIDIA GPU and a compatible NVIDIA driver. The
reference machine has an A100-SXM4-40GB and driver 595.91.07. TimesFM, Toto, and
Moirai use CUDA 13.0 Torch binaries; the Chronos baseline uses CUDA 12.8. Setup
does not install or change drivers. Model downloads need network access and disk
space; keep weights outside Git.

```bash
./scripts/setup.sh
./scripts/doctor.sh
./scripts/serve.sh --adapter chronos2 --backend vtsfm --port 8001
```

From another terminal in this checkout:

```bash
./scripts/request.sh http://127.0.0.1:8001
```

The [example request](examples/request.json) contains clearly identified
synthetic numerical smoke-test inputs. It is separate from benchmark sensor
data. Serving uses `POST /pooling` and binds to loopback by default.

Choose one backend per server process:

```bash
./scripts/serve.sh --backend chronos2 --port 8001
./scripts/serve.sh --backend native --port 8001
```

`--backend upstream` selects the chosen adapter's upstream plugin; `chronos2`
remains an equivalent name for the first adapter's comparator.

Stop the previous server before starting another. `vtsfm` selects this project's
adapter, `chronos2` selects the pinned community adapter, and `native` uses the
official Chronos pipeline behind the same HTTP request/response boundary.
`--adapter` selects the model family independently of `--backend`; it defaults to
`chronos2`, so existing commands keep their behavior. The additional adapters
use the smaller numerical contract described below.

## TimesFM 2.5, Toto 2, and Moirai 2.0

These adapters use the same vLLM pooling interface and preserve upstream forecast
semantics. Whole-forecast batching and TimesFM block scheduling are described
below; attention remains upstream-owned. Each has its own environment, leaving
`.venv` and the Chronos baseline unchanged:

```bash
./scripts/setup.sh --adapter timesfm25
./scripts/serve.sh --adapter timesfm25 --backend vtsfm --port 8001
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @examples/timesfm25.json http://127.0.0.1:8001/pooling
```

Replace `timesfm25` with `toto2` or `moirai2` for the other adapters. These use
`--backend vtsfm`; the serving CLI's `--backend native` and the older
`benchmarks.run` comparator remain Chronos-specific. The manual performance
profile below provides a native HTTP baseline for these three adapters.

Inputs are finite univariate `target` arrays with optional `item_id`; separate
inputs are independent series. Parameters are `prediction_length` (1–1024) and
`quantile_levels` (unique increasing tenths from 0.1 through 0.9). Responses contain
the requested quantiles and item IDs, without a synthetic `mean` field. Covariates,
cross-learning, and multivariate targets are rejected. Context limits are 8192 for
TimesFM/Toto and 7168 for Moirai, reserving room for the prediction horizon.

TimesFM keeps histories and forecast tensors on the model device and uses the
pinned upstream decoder. It preserves the reference normalization, continuous
quantile head, flip invariance, positivity, and crossing correction.
Toto uses the documented 768-point block decoding setting. Moirai keeps
histories and forecasts on the model device and calls the pinned upstream
`Moirai2Forecast.forward`, including its recursive quantile forecast. Shorter
histories in a request are masked as padding, so each forecasts as it would alone.

The shared IO also supports sequential offline `LLM.encode({'data': payload},
pooling_task='plugin')` calls. See [request lifecycle](docs/request-lifecycle.md)
for supported inputs, concurrent serving, cancellation, and the manual TimesFM
lifecycle suite. Response metadata travels with each result; IO keeps no pending
request map.

TimesFM now enforces a measured numerical envelope: at most 16 series per
request, context 8192, horizon 1024, and four scheduled requests. The worker
splits compatible work within execution budgets. Use `--forecast-limits FILE`
to tighten the policy; see [numerical admission](docs/numerical-admission.md) for
defaults, errors, memory measurements, and the manual GPU suite. Toto and Moirai
also have model-specific policies and bounded whole-forecast batching, qualified
on the reference A100. Request caps are 16 series for Toto and eight for Moirai,
with at most four scheduled requests. These finite probes establish the tested
envelope; they do not establish a throughput speedup.

TimesFM also supports experimental forecast-block continuous batching:

```bash
./scripts/serve.sh --adapter timesfm25 --continuous-batching --port 8001
```

Requests can join and finish between 128-point forecast blocks. The default is
four slots, bounded active numerical work, and 16 waiting/running requests per
API process (overflow: HTTP 503). Final responses are unchanged. See
[continuous batching](docs/continuous-batching.md) for adapter-owned caches, the
pinned V1 scheduler/worker hooks, parity tests, memory measurements and offered-load
benchmarks. This mode is opt-in.

Toto uses Python 3.12.12 and a pinned upstream revision. Moirai keeps its
NumPy 1.26 dependencies and explicitly overrides its Torch
upper bound and vLLM's OpenCV requirement in `scripts/moirai2-overrides.txt`.
These overrides apply only to `.venv-moirai2`; OpenCV vision processing is unused.

If upgrading an existing CUDA 12 model environment, move it aside before
running setup (for example, `mv .venv-timesfm25 .venv-timesfm25-cu128`), or use a
fresh checkout. The CUDA 12 and CUDA 13 wheels can own overlapping files.

Check checkpoint loading, multimodal IO, and parity against the official forecast
on CPU inside a model environment (replace `timesfm25` for the other models):

```bash
CUDA_VISIBLE_DEVICES='' VTSFM_TEST_ADAPTER=timesfm25 \
  .venv-timesfm25/bin/python -m pytest tests/test_forecast_runtime.py -q
```

This downloads the pinned checkpoint. GPU HTTP smoke tests passed for all three
on an RTX 4080 Laptop GPU using cached checkpoints, FP32, eager execution, and
`--gpu-memory-utilization 0.35`. Checks covered examples, unequal history lengths,
and repeated requests; they do not establish performance or forecast accuracy.

For a manual native-GPU versus HTTP parity run across all three adapters, use
`./scripts/test-models.sh`. It includes synthetic and pinned AssetOpsBench cases;
`--profile accuracy` adds observed-label scoring, and `--profile performance`
compares matching native/vLLM HTTP workloads. See the [model integration guide](docs/model-integration.md)
for environment setup, idle-GPU requirements, per-model selection, and saved
pre-merge evidence.

## Run a comparison

Benchmarks require exclusive access to the selected GPU and a free port. The
runner refuses occupied resources, launches its own servers sequentially, and
cleans up only children it owns. It never stops another person's service. Stop
your manually launched server before starting the benchmark, or use a separate
idle GPU machine.

The standard input is the pinned AssetOpsBench MetroPT compressor fixture:

```bash
.venv/bin/python -m benchmarks.prepare_data --output data
./scripts/smoke.sh --backends native vtsfm --port 8001 \
  --data-dir data --output results/smoke-vtsfm
```

Alternatively, `./scripts/smoke.sh --synthetic --output results/synthetic-smoke`
checks the serving path without downloading sensor data. Synthetic inputs are
permitted only for smoke tests and are labeled in run metadata.

For the reference grid and sample counts:

```bash
.venv/bin/python -m benchmarks.run \
  --backends native vtsfm --port 8001 --data-dir data \
  --batches 1 4 16 64 128 --contexts 128 512 2048 4096 \
  --blocks 3 --repetitions 50 --warmups 10 \
  --output results/native-vtsfm
.venv/bin/python -m benchmarks.report \
  --results results/native-vtsfm --output results/native-vtsfm/report
```

Replace `vtsfm` with `chronos2` to measure the community comparator. Each run
compares native inference with one selected plugin. Save each run to a new output
directory. Raw timing samples, payload hashes, configuration, numerical parity
checks, summaries, and GPU memory measurements make comparisons auditable.
Reports retain failed and incomplete configurations.

The current runner sends sequential requests containing batches of univariate
sensor/history windows, with `cross_learning=false`. It does not benchmark live
cross-request batching, grouped or covariate workloads, or concurrent clients.

The fixture has seven analog channels over 5,116 hourly grid positions; 560 rows
are wholly missing. Preparation causally forward-fills those rows, preserves
observed values before conversion to FP32, and records source checksums and every
imputed index. Windows overlap and share one compressor, so a batch of 128 means
128 correlated sensor/history windows. These runs measure inference performance;
they do not establish forecasting accuracy. Context 8,192 is unsupported by this
fixture. See [data and measurement details](docs/architecture.md#benchmark-contract).

## Reproducible runtime pins

| Adapter | vLLM | Torch | Torch CUDA runtime | Transformers |
| --- | --- | --- | --- | --- |
| TimesFM 2.5, Toto 2, Moirai 2.0 | 0.33.0 (V1 runner) | 2.14.0 | 13.0 | 5.20.2 |
| Chronos-2 baseline | 0.16.0 (V1 runner) | 2.9.1 | 12.8 | 4.57.6 |

The vLLM release determines the compatible Torch and CUDA binaries. The
forecasting launcher selects V1 because V2 does not support the `plugin` task.
See the [runtime upgrade spike](docs/runtime-upgrade.md) for the A100 results.

The remaining settings for the Chronos-2 baseline are:

| Component | Pin / setting |
| --- | --- |
| Python | 3.11.17 |
| Chronos integration | PR [#473](https://github.com/amazon-science/chronos-forecasting/pull/473), commit `ba47d25a04aedac33af2fb9780609afd54744d5e` |
| Model | `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| Execution | FP32, eager; `cross_learning=false`; one outstanding HTTP request |

These pins define a research baseline. A dependency or precision change requires
a separate result set. CPU checks do not verify GPU compatibility or performance.
See the [starter verification record](docs/verification.md) for the initial CPU
and GPU checks.

## Extension boundaries

| Area | Entry points | Responsibility |
| --- | --- | --- |
| Adapter metadata and discovery | `src/vtsfm/adapters/base.py`, `src/vtsfm/adapters/registry.py` | Describe and resolve model families without importing runtimes |
| Chronos-2 model and numerical IO | `src/vtsfm/adapters/chronos2/` | Own model pins, preparation, group semantics, native comparison, and numerical parity |
| Shared batching | `src/vtsfm/batching.py` | Plan compatible prepared rows and route results within adapter-specific limits |
| Forecast-block scheduling | `src/vtsfm/stepping.py`, `continuous_scheduler.py`, `block_worker.py`, `block_pooling.py` | Schedule resumable TimesFM forecasts with request-owned state and bounded intake |
| Plugin integration | `src/vtsfm/plugin.py`, `src/vtsfm/cli.py` | Register lazy targets and select adapter/backend settings |
| Numerical transport and output routing | `src/vtsfm/adapters/multimodal.py`, `transport.py`, `pooling.py`, `src/vtsfm/worker.py` | Preserve request identity and crop per-request forecasts in actual worker slot order |
| Kernel research | Adapter model attention boundary | Profile and implement family-specific kernels; not implemented yet |
| Measurement | `benchmarks/`, `src/vtsfm/benchmark_memory.py` | Reproduce native/community/vTSFM comparisons and review raw samples |

The former model, IO, and native module paths remain compatibility shims. To add a
model family, provide its metadata and implementations, register the descriptor,
and validate its preparation and output contract; see
[adding an adapter](CONTRIBUTING.md#adding-an-adapter).

For serving or performance changes, run relevant CPU checks, reproduce the
untouched GPU smoke comparison on an idle machine when available, and save a
baseline before optimizing.
Assign each proposed optimization a numerical check and a measurable target.
Read [architecture](docs/architecture.md) and [contribution guidance](CONTRIBUTING.md)
before changing plugin discovery or the serving contract.
See [numerical transport](docs/numerical-transport.md) for the pinned vLLM worker
boundary and the manual checkpoint-free multi-request probe.
TimesFM, Toto, and Moirai expose scheduled concurrency through
`serve --max-num-seqs 3`; the default is one. For experimental whole-forecast
whole-forecast batching, combine `--max-num-seqs 4` with
`--forecast-batch-rows 8`. See the [batching boundaries](docs/numerical-transport.md#current-execution-boundary).

Agents should start with [AGENTS.md](AGENTS.md); `CLAUDE.md` imports the same
conventions. GitHub Actions runs basic Python lint, CPU tests, shell syntax
checks, and package build/install smoke checks.
Full-model GPU validation, accuracy/performance benchmarks, and deployment remain manual.

## License

Project code is licensed under [Apache-2.0](LICENSE). Upstream software and
datasets retain their own terms and attribution; see
[third-party notices](THIRD_PARTY_NOTICES.md). Downloaded sensor data, weights,
virtual environments, and generated run artifacts are not committed.
