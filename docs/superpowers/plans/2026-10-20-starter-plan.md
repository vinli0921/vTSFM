# vTSFM Starter Implementation Plan

> **For agentic workers:** Use parallel scoped implementation and verification.

**Goal:** A runnable, tested, cloneable starter for five developers.

**Architecture:** External vLLM plugin with lazy registration and thin inherited
Chronos adapters, pinned optional GPU dependencies, portable owned-process
benchmarking, and CPU-first onboarding and CI.

**Tech Stack:** Python 3.11, uv, vLLM 0.16, Torch 2.9.1, FastAPI, NumPy, pytest.

**Spec:** ../specs/2026-10-20-starter-design.md

## Global constraints

Preserve the verified upstream installation. Use plugin name `vtsfm` for both
entry-point groups and community comparator `chronos2`. GPU environment is a
per-clone `.venv`; CPU mode installs `.[dev,benchmark]` and GPU mode installs
`.[serve,dev,benchmark]`. No hardcoded usernames, addresses, or installation paths.

## Tasks

- [x] Package/plugin: pyproject, pins, lazy discovery, inherited model/IO,
  `python -m vtsfm.cli`, CPU import tests and command tests.
- [x] Portable benchmark: adapt the verified harness and data preparer; test
  metrics, payload identity, occupied-resource refusal, partial reports and cleanup.
- [x] Setup/scripts: repeatable CPU/GPU setup, doctor, serving, requests, smoke;
  shell syntax and useful failure behavior; no driver modifications.
- [x] Documentation/CI: team quick start, architecture, tasks, licenses, CPU CI,
  numerical example request and clearly labeled historical baseline CSV.
- [x] Integrate and test: full CPU tests, packaging, isolated GPU smoke comparing
  native/community/vtsfm, independent review and final Git/secret/artifact audit.
- [ ] Initialize and commit repository, then publish to the user's destination
  when provided; verify a fresh clone and clean working tree.

## Shared interfaces

`vtsfm.config`: MODEL_ID, MODEL_REVISION, CHRONOS_COMMIT, VLLM_VERSION,
TORCH_VERSION, TRANSFORMERS_VERSION, PYTHON_VERSION, DEFAULT_PORT=8001.

`vtsfm.cli.build_vllm_command(backend='vtsfm', host='127.0.0.1',
port=8001, benchmark=False) -> (list[str], dict[str,str])` constructs a
command and complete child environment for either `vtsfm` or `chronos2`.
Benchmark mode alone enables development RPC and attaches
`vtsfm.benchmark_memory.MemoryMetrics`. CLI `serve --backend
{vtsfm,chronos2,native} --host --port [--benchmark] [--dry-run]`.

`benchmarks/run.py`: `--smoke --output PATH --port 8001 --backends native vtsfm`
plus explicit grid/repetition options. Engine labels native/vllm may be kept in
raw records for compatibility, with actual plugin name recorded in config;
comparison is native versus one selected plugin per run. Never stop unowned
processes. Use existing data-preparation and reporting artifact formats.

`scripts/setup.sh [--cpu]`, `scripts/doctor.sh`, `scripts/serve.sh [serve args]`,
`scripts/request.sh [URL]`, `scripts/smoke.sh [benchmark args]`.

## Verification focus

CPU import must not initialize CUDA. Model subclass needs its own multimodal
registration. Select vtsfm explicitly to avoid the upstream entry point
overwriting it. Resource refusal and cancellation must release only owned
children. Failures and absent samples remain visible in reports; output
equivalence is checked outside timed intervals. Defaults are reversible and
all setup paths resolve relative to the clone.
