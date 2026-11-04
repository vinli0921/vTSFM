# Agent conventions for vTSFM

vTSFM is an early-stage research extension for vLLM. Favor small working
experiments, short feedback loops, and clear code. Use judgment: routine changes
do not need an RFC, written plan, new abstraction, or an approval ceremony.

## Start here

- Read [README.md](README.md) for setup and current support. Use
  [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md)
  when touching adapter interfaces, serving, or benchmarks.
- Check the current branch, local changes, and nearby code before editing.
  Preserve teammates' work and check for overlapping work when relevant.
- Keep changes focused. Label experimental behavior and explain shortcuts that
  affect correctness or measurements. Avoid unrelated cleanup and speculative
  frameworks for future models.

## Code and architecture

- `src/vtsfm/adapters/<family>/` owns model-specific preparation, inference,
  numerical IO, and pins. Shared batching and plugin code should stay reusable
  across model families.
- Keep adapter discovery, CLI help, and dry-runs CPU-safe. Import Torch, vLLM,
  and model packages only where runtime execution needs them.
- Preserve normalization, padding/masks, independent series or attention groups,
  horizons, and probabilistic output semantics when changing execution paths.
  Document tensor shapes and device expectations at shared boundaries.
- Match nearby style. Prefer clear names, useful interface types, and short
  comments explaining non-obvious behavior. Formatting and strict typing are
  not gates for this research project.
- Keep dependency/checkpoint pins consistent with adapter metadata. Changing
  pins or defaults is fine when intentional; explain the change and record a
  separate baseline when it affects a comparison.

## Local setup and checks

Use the checkout's environments, not global Python packages:

```bash
./scripts/setup.sh --cpu
./scripts/doctor.sh --cpu
.venv/bin/python -m pytest tests/test_batching.py -q  # example targeted check
.venv/bin/python -m pytest -m 'not gpu'             # broader CPU checks
```

- GPU setup targets Linux. Use `./scripts/setup.sh --adapter timesfm25`,
  `toto2`, or `moirai2` and the corresponding `.venv-<adapter>/bin/python`.
- Run the smallest useful checks for the change. Shared behavior changes usually
  warrant broader CPU tests; docs-only changes need a read-through and link check.
  For shell edits, run `bash -n` on the affected scripts.
- Add regression coverage when it helps catch a real failure. Reuse existing
  tests and fixtures; avoid tests that only restate the implementation.
- Serving/model changes need relevant HTTP smoke and numerical parity checks
  when a suitable environment is available. Say what ran, what failed, and what
  remains unverified; CPU tests do not establish GPU compatibility.
- There is no CI/CD pipeline or mandatory pre-commit gate. Keep validation local
  unless the team explicitly asks to introduce automation.

## Experiments and shared resources

- Save a baseline before optimizing. Compare matching workloads and settings;
  record revisions, hardware, precision, warmups, timing boundaries, failures,
  and memory methodology. Distinguish observed, imputed, and synthetic data.
- Describe only what is implemented and measured. The shared batching planner
  is currently metadata-only; the newer adapters run whole upstream forecasts.
  Neither alone establishes live cross-request batching or an attention speedup.
- Use an idle GPU and free port for benchmarks. Do not interrupt another
  person's workload or change shared drivers/services without authorization.
  Clean up only processes and artifacts owned by the experiment.
- Keep credentials, checkpoints, raw downloads, environments, and generated runs
  out of Git. Preserve third-party attribution.

Keep handoffs short: what changed, why, checks run, and remaining limitations.
These conventions take inspiration from [vLLM's agent guide](https://github.com/vllm-project/vllm/blob/main/AGENTS.md),
adapted to a small research team with lightweight local validation.
