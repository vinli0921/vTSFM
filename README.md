# vTSFM

A team starter for serving time-series forecasts through an external vLLM
plugin. The `vtsfm` plugin wraps the existing Chronos-2 integration and provides
separate model and IO entry points for development. The original `chronos2`
plugin remains available as a comparator.

This starter does not implement new batching or attention optimizations.
Cross-request packing and grouped attention kernels are future work. The
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
.venv/bin/python -m benchmarks.run --help
```

## Serve on a GPU

Use a Linux machine with an NVIDIA GPU and a compatible NVIDIA driver. The
verified reference machine has an A100-SXM4-40GB, CUDA 12.8 Torch binaries, and
driver 595.91.07. Setup does not install or change drivers. Model downloads need
network access and disk space; keep weights outside Git.

```bash
./scripts/setup.sh
./scripts/doctor.sh
./scripts/serve.sh --backend vtsfm --port 8001
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

Stop the previous server before starting another. `vtsfm` selects this project's
adapter, `chronos2` selects the pinned community adapter, and `native` uses the
official Chronos pipeline behind the same HTTP request/response boundary.

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

The fixture has seven analog channels over 5,116 hourly grid positions; 560 rows
are wholly missing. Preparation causally forward-fills those rows, preserves
observed values before conversion to FP32, and records source checksums and every
imputed index. Windows overlap and share one compressor, so a batch of 128 means
128 correlated sensor/history windows. These runs measure inference performance;
they do not establish forecasting accuracy. Context 8,192 is unsupported by this
fixture. See [data and measurement details](docs/architecture.md#benchmark-contract).

## Reproducible baseline

| Component | Pin / setting |
| --- | --- |
| Python | 3.11.17 |
| Chronos integration | PR [#473](https://github.com/amazon-science/chronos-forecasting/pull/473), commit `ba47d25a04aedac33af2fb9780609afd54744d5e` |
| Model | `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| vLLM | 0.16.0 |
| Torch | 2.9.1 |
| Transformers | 4.57.6 |
| Execution | FP32, eager; `cross_learning=false`; one outstanding HTTP request |

These pins define a research baseline. A dependency or precision change requires
a separate result set. CPU CI does not verify GPU compatibility or performance.
See the [starter verification record](docs/verification.md) for the initial CPU
and GPU checks.

## Where to work first

| Area | Entry points | Initial responsibility |
| --- | --- | --- |
| Model integration | `src/vtsfm/plugin.py`, `src/vtsfm/models/chronos2.py` | Keep lazy discovery and numerical parity intact |
| Numerical IO | `src/vtsfm/io_processor.py` | Preserve input grouping, horizon, and quantile output schema |
| Scheduling research | vLLM/plugin integration boundaries | Design cross-request packing; not implemented yet |
| Kernel research | model attention boundary | Profile and propose grouped kernels; not implemented yet |
| Measurement | `benchmarks/`, `src/vtsfm/benchmark_memory.py` | Reproduce native/community/vTSFM comparisons and review raw samples |

During the first week, have everyone run CPU checks, reproduce the untouched GPU
smoke comparison on an idle machine, and save a baseline before changing behavior.
Assign each proposed optimization a numerical check and a measurable target.
Read [architecture](docs/architecture.md) and [contribution guidance](CONTRIBUTING.md)
before changing plugin discovery or the serving contract.

## License

Project code is licensed under [Apache-2.0](LICENSE). Upstream software and
datasets retain their own terms and attribution; see
[third-party notices](THIRD_PARTY_NOTICES.md). Downloaded sensor data, weights,
virtual environments, and generated run artifacts are not committed.
