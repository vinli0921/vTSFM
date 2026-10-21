# vTSFM team starter

Build a cloneable Python repository for the team's already-selected external
vLLM plugin project. The runnable skeleton delegates to Chronos PR #473 at
ba47d25a04aedac33af2fb9780609afd54744d5e and registers its own `vtsfm` model and
IO entry points. It is an integration starting point, with no optimization or
novelty claim. Keep the installed community plugin available as a comparator.

Use Python 3.11.17, vLLM 0.16.0, Torch 2.9.1, Transformers 4.57.6,
Chronos model revision 29ec3766d36d6f73f0696f85560a422f50e8498c,
FP32 and eager execution. GPU setup is per-checkout; CPU setup supports tests,
data preparation and report generation without installing Torch/vLLM.

Include setup, doctor, serve, request and smoke scripts; portable numerical
benchmarking against native HTTP and the community plugin; explicit data
provenance; raw latencies, pooled p95, series/s, and GPU memory; plotting;
CPU CI; tests; contribution guidance and upstream attribution. The baseline
fixture is pinned AssetOpsBench hourly MetroPT, with documented causal forward
fill. Synthetic inputs may be used only for clearly identified smoke tests.

The harness owns only processes it launches, rejects occupied ports and GPUs,
and never stops an existing service. The current VM's baseline remains intact.
Validate in an isolated checkout/environment on that VM and restore its running
baseline after any explicitly managed exclusive test. Do not change drivers.

Publish only after the repo is concrete and tested. The user supplied
https://github.com/vinli0921/vTSFM.git as the destination. Use the stated private
visibility default. No team invitations are inferred.

## Boundaries

- `src/vtsfm/config.py`: pure constants and environment pins.
- `src/vtsfm/plugin.py`: lazy model registration and IO factory.
- `src/vtsfm/models/chronos2.py`, `io_processor.py`: thin upstream subclasses,
  with explicit multimodal processor registration on the new model class.
- `src/vtsfm/cli.py`: command construction and process exec; no CUDA import for help.
- `src/vtsfm/native_server.py`, `benchmark_memory.py`: native comparator and counters.
- `benchmarks/`: portable runner, data preparation, common metrics and reports.
- `scripts/`: per-clone setup, environment checks, serving and smoke entry points.
- `tests/`: meaningful CPU tests plus opt-in GPU integration checks.

Do not implement cross-request packing, grouped attention kernels, TimesFM,
MCP integration, or automatic driver changes in this starter. Document them as
future tasks rather than advertise them as implemented.
