# Starter verification — 2026-10-20

The initial starter was checked before publication with the pinned dependencies
in [the README](../README.md#reproducible-baseline).

- CPU tests: 34 passed. Coverage includes lazy plugin discovery, command
  construction, script boundaries, data provenance, numerical validation,
  resource ownership, cancellation, and incomplete reports.
- Setup: a fresh CPU checkout with spaces in its path installed Python 3.11.17;
  environment checks and repeat setup passed without replacing global tools.
  An isolated Linux GPU checkout installed the serving dependencies and passed
  CUDA environment checks on an A100-SXM4-40GB.
- Publication: a fresh clone from GitHub completed the documented CPU setup,
  environment checks, and all 34 tests on macOS. The first [GitHub Actions run](https://github.com/vinli0921/vTSFM/actions/runs/39532045316)
  also passed the same setup, CPU tests, and shell syntax checks on Ubuntu.
- Packaging: the wheel built successfully and contained the model adapter and
  both vLLM plugin entry points.
- GPU integration: separate `native vtsfm` and `native chronos2` smoke comparisons
  completed using the prepared AssetOpsBench fixture, batch size 2, context 128,
  and prediction length 96. Each comparison recorded three timed requests per
  backend, with numerical checks for every response. Maximum absolute error
  against native inference was `1.430511474609375e-06`.
- Benchmark lifecycle: both comparisons released their owned server processes;
  the pre-existing service was restored and passed health and forecast checks.
  The vTSFM smoke report rendered with two complete cells and no warnings.

These small smoke runs establish integration and numerical parity. Their sample
counts do not establish performance gains. The [historical baseline](baselines/2026-10-20/README.md)
is a separate, larger experiment measuring the community plugin. New attention
kernels, cross-request batching, and AssetOpsBench MCP integration remain future
work.

## Adapter refactor — 2026-10-23

- All 150 CPU tests passed on Python 3.11.17, including the existing suite and
  new adapter selection, group planning/routing, and source snapshot checks.
- The wheel built with all adapter modules and the shared planner. A fresh
  wheel-only environment loaded both entry points, listed adapters, planned and
  routed example groups, and constructed the serving command without GPU packages.
- Independent code review found no important issues. Native forecasting functions
  were preserved during relocation; launch settings and model pins are unchanged.
- The feature branch's push and pull-request GitHub Actions checks passed,
  including fresh CPU setup and the full test suite.
- GPU startup was not rerun because the existing VM did not respond to SSH.
  The earlier GPU smoke results above apply to the original starter, not a new
  GPU verification of this refactor. No remote service was interrupted.

Shared batching in this version plans metadata only. It is not connected to
vLLM execution and introduces no measured throughput or latency improvement.
