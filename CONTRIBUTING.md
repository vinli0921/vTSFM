# Contributing

Agent conventions live in [AGENTS.md](AGENTS.md); `CLAUDE.md` imports that guide.
This is an early-stage research project: use focused changes and proportionate
local checks. GitHub Actions runs basic lint, CPU tests, and package checks;
the optional Modal workflow adds small CUDA contract checks on relevant trusted
pull requests. Full-model validation, performance benchmarks, and deployment
remain manual. There is no required pre-commit gate.

Start with the [architecture guide](docs/architecture.md).
For tensor adapters and shared worker work, use the
[prepared execution contract](docs/execution-contract.md) and its CPU example.
Chronos-2 delegates to the pinned Chronos integration. TimesFM, Toto, and Moirai
use whole-forecast pooling wrappers. Shared batching is a metadata planner outside
live serving; do not describe an optimization as implemented until code,
numerical checks, and measurements exist.

## Local checks

```bash
./scripts/setup.sh --cpu
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -m 'not gpu'
for script in scripts/*.sh; do bash -n "$script"; done
.venv/bin/python -m build
```

Lint catches basic Python errors; there is no formatting gate. Package validation
builds an sdist and then a wheel from it, installs the wheel into a fresh
environment without optional dependencies, and checks the installed CLI, both
vLLM plugin entry points, and adapter module discovery outside the checkout.

CPU checks must work without Torch, vLLM, an NVIDIA GPU, or model downloads.
Keep imports lazy: package discovery and CLI help must not initialize CUDA.
GPU integration checks are opt-in and run separately on an idle Linux GPU host.
The [Modal GPU workflow](docs/gpu-testing.md) runs a small CUDA contract suite on
relevant same-repository pull requests and on demand; its guide covers account
setup, secrets, and credit-only billing limits.
CPU checks do not substitute for numerical and serving validation.

## Change boundaries

- Keep model/revision pins in the adapter descriptor, compatibility aliases in
  `src/vtsfm/config.py`, and dependency pins in package metadata consistent. Record
  any intentional baseline change.
- Register the `vtsfm` model and IO entry points explicitly. The model subclass
  needs its own multimodal processor registration; inheritance alone is
  insufficient. Select one plugin explicitly when serving.
- Preserve batch grouping, `cross_learning`, prediction horizon, quantiles, and
  response schema when comparing backends. Check numerical equivalence outside
  timed intervals.
- Add meaningful tests for behavior changes. Avoid tests that merely duplicate
  the implementation. Run relevant CPU tests before requesting review.
- Keep scripts relative to the checkout and use its `.venv`. Do not add usernames,
  machine addresses, global environment mutations, or driver changes.

## Adding an adapter

1. Create `src/vtsfm/adapters/<family>/` with a pure metadata `AdapterSpec` in
   `spec.py` and lazy model/IO implementations. Supply a name, exact model ID and
   revision, architecture names, importable model and IO class paths, upstream IO
   plugin, native server module, vLLM arguments, and an honest group-attention
   capability flag. Keep model-family pins and execution settings here.
   Declare and pin required optional runtime dependencies in package metadata and
   the setup extras before claiming that adapter can be installed and served.
2. Add the descriptor to `adapters/registry.py`. Discovery, `list_adapters()`,
   `get_adapter(name)`, CLI help, `adapters`, and serving dry-runs must work without
   importing Torch, vLLM, or model packages. Keep both `vtsfm` plugin entry points
   stable; plugin registration and IO selection must follow descriptor metadata.
3. Implement the family's normalization, patching, covariates, group semantics,
   output decoding, and native comparator inside its adapter. Document the
   numerical HTTP contract. The current Chronos-2 schema is not a common public
   schema for other models. Add required concrete-class processor registration
   rather than assuming inheritance handles it.
4. Define how prepared rows map to `BatchKey` and `BatchRequest.group_sizes`.
   Declare `execution_capabilities` and relevant `execution_options`, and follow
   the prepared input/output shapes when implementing `ForecastExecutor`.
   Preserve independent requests and cross-learning semantics. Keep the shared
   planner independent of model imports; model-specific target/covariate filtering
   and kernels belong in the adapter. The planner is not yet wired into serving,
   so planning tests alone do not validate execution or performance.
5. Verify CPU-safe discovery and selection, invalid-name failures, preparation and
   output semantics, and numerical parity on an idle GPU with matching native
   inputs. Record model-specific launch requirements and benchmark limitations.
   Adding metadata alone does not make a model supported.

Preserve the existing Chronos-2 defaults, compatibility import paths, and
`MODEL_ID`, `MODEL_REVISION`, and `CHRONOS_COMMIT` aliases. Another adapter must not
silently change the pinned Chronos-2 baseline or imply the benchmark runner can
evaluate that model without further work.

## Performance changes

Save an untouched baseline and run the same payloads, settings, GPU, and timing
boundary before and after a change. Include raw samples, exact source/model
revisions, warmups, repetitions, precision, failures, and memory methodology.
Report end-to-end HTTP latency and series/s with one outstanding request unless a
separate concurrency experiment is explicitly specified. Do not call native-only
model timings a matched comparison to the HTTP path.

The current workload is sequential requests of univariate sensor/history windows
with `cross_learning=false`. It does not validate grouped attention, covariates,
concurrent clients, or live cross-request batching. Add separate workload and
correctness coverage before making claims about those behaviors. A metadata
planner change alone is not a GPU optimization or a measured speedup.

Use a free GPU and port. The runner must refuse foreign GPU workloads or an
occupied port and clean up only processes it launched. Never stop an existing
service to make a benchmark pass. Test cancellation and partial-run reporting
when changing process management.

Describe whether data is observed, imputed, or synthetic. The reference fixture's
560 missing hourly rows use causal forward-fill; overlapping sensor/history
windows are correlated. Synthetic inputs belong only in labeled smoke tests.
Do not infer forecast accuracy from these performance experiments.

## Review and repository hygiene

Keep pull requests focused. Explain the concrete trigger and resulting behavior,
then list relevant validation and limitations. Call out changed pins, IO semantics,
measurement boundaries, or experimental settings so reviewers can assess them.

Do not commit model weights, raw sensor downloads, credentials, virtual
environments, caches, logs, or generated runs. A small reviewed historical summary
may live under `docs/baselines/` with provenance and explicit backend labels.
Preserve upstream license headers and attribution. Project contributions use
Apache-2.0; third-party data retains its original license.
