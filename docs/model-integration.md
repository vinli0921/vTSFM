# Manual GPU model integration checks

Run this suite on a Linux NVIDIA GPU host before merging a model or serving
change. It compares the pinned TimesFM 2.5, Toto 2, and Moirai 2.0 native GPU
forecasting paths with the vTSFM HTTP path. Inputs are synthetic correctness
fixtures, not AssetOpsBench accuracy or performance measurements.

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

Each model receives four requests, in order:

1. One 128-point history, a 17-point forecast, and quantiles 0.1/0.5/0.9.
2. Two unequal histories of 97 and 163 points, a 33-point forecast, and only
   quantiles 0.2/0.8, checking padding, item IDs, and channel selection.
3. A longer forecast: 256 points for TimesFM, 800 for Toto, and 160 for Moirai,
   with all nine quantiles. Toto's case exceeds its 768-point decoding block.
4. The first request repeated after the horizon and batch shape changes.

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
