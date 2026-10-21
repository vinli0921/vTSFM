# Contributing

Start from the approved [starter design](docs/superpowers/specs/2026-10-20-starter-design.md)
and [implementation plan](docs/superpowers/plans/2026-10-20-starter-plan.md).
The starter delegates to the pinned Chronos integration; do not describe an
optimization as implemented until code, numerical checks, and measurements exist.

## Local checks

```bash
./scripts/setup.sh --cpu
.venv/bin/python -m pytest -m 'not gpu'
for script in scripts/*.sh; do bash -n "$script"; done
```

CPU checks must work without Torch, vLLM, an NVIDIA GPU, or model downloads.
Keep imports lazy: package discovery and CLI help must not initialize CUDA.
GPU integration checks are opt-in and run separately on an idle Linux GPU host.
CI's CPU checks do not substitute for numerical and serving validation.

## Change boundaries

- Keep model/revision and dependency pins in `src/vtsfm/config.py` and package
  metadata consistent. Record any intentional baseline change.
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

## Performance changes

Save an untouched baseline and run the same payloads, settings, GPU, and timing
boundary before and after a change. Include raw samples, exact source/model
revisions, warmups, repetitions, precision, failures, and memory methodology.
Report end-to-end HTTP latency and series/s with one outstanding request unless a
separate concurrency experiment is explicitly specified. Do not call native-only
model timings a matched comparison to the HTTP path.

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
