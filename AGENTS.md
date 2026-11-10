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
- Keep GitHub Actions lightweight: basic lint, CPU tests, and package checks.
  The enabled Modal workflow adds small CUDA contract checks on relevant trusted
  pull requests. Full-model GPU validation, performance benchmarks, and deployment
  remain manual unless the team explicitly asks to automate them. There is no
  mandatory pre-commit gate.

## Modal CUDA contract checks

- Follow [docs/gpu-testing.md](docs/gpu-testing.md). Automatic runs use the changed
  paths in `.github/workflows/gpu-tests.yml` and accept only same-repository PRs,
  including drafts. Skip forks, never use `pull_request_target` with credentials,
  and do not add a duplicate GPU `push` trigger. PR path filters consider the
  whole PR diff, so later documentation-only commits can still match.
- `MODAL_GPU_ENABLED=true` is the owner's persistent acknowledgement of the
  workspace credit allowance, usage budget, and $0 out-of-pocket spend limit.
  Keep both Modal token values in GitHub secrets. The flag and timeouts do not
  inspect live billing or guarantee free execution; unset/false disables submission.
- Manual dispatch overrides the path filter and also requires its billing-limit
  checkbox; it does not override disabled setup. Keep the single global
  concurrency group with `cancel-in-progress: false` so an active GPU app is not
  orphaned by a newer commit.
- Preserve the small job's bounds: one GPU (L4 preferred, T4 fallback), one CPU,
  4096 MiB memory, one single-use container, no warm pool/volumes/retries,
  120-second startup, 300-second execution,
  240-second test timeout, and a 30-minute GitHub job. This suite checks CUDA tensor
  contracts and routing; it does not establish model parity, accuracy, or performance.
- Record the running app ID in a fresh private file before submitting GPU work.
  Always clean up only that validated ID, never an app name or unrelated workspace
  resources. No recorded ID means no cleanup API call; after forced cancellation,
  verify that run's app has stopped. Keep credentials off the remote test worker.

## Experiments and shared resources

- Save a baseline before optimizing. Compare matching workloads and settings;
  record revisions, hardware, precision, warmups, timing boundaries, failures,
  and memory methodology. Distinguish observed, imputed, and synthetic data.
- Describe only what is implemented and measured. Whole-forecast batching uses
  each adapter's own cost policy. TimesFM has a measured numerical envelope;
  Toto/Moirai candidate caps still require the documented idle-GPU validation.
  TimesFM can opt into forecast-block
  scheduling; no new attention kernel or vLLM-managed KV is claimed.
- Use an idle GPU and free port for benchmarks. Do not interrupt another
  person's workload or change shared drivers/services without authorization.
  Clean up only processes and artifacts owned by the experiment.
- Keep credentials, checkpoints, raw downloads, environments, and generated runs
  out of Git. Preserve third-party attribution.

## Manual GPU model validation

- Run `scripts/test-models.sh` on an active, idle Linux GPU node; it needs no
  stable IP, SSH secret, Actions runner, or automatic VM startup. Each adapter
  uses `.venv-timesfm25`, `.venv-toto2`, or `.venv-moirai2`. See
  [the model validation guide](docs/model-integration.md).
- Before merging serving/model changes, use `--profile integration` for actual
  vLLM HTTP versus independent native CUDA parity. Synthetic edge cases and a
  small pinned AssetOpsBench slice cover different failure modes. The first run
  prepares the documented sensor fixture; keep downloaded data outside Git.
- Use `--profile accuracy` for model, precision, or approximate-execution changes.
  Optional prepared fev slices are custom univariate smoke tasks, not full
  benchmark submissions. Score observed future labels only; fit no scales from
  holdouts. MAE/MASE use the median, SQL uses all nine quantiles. Undefined scales
  must remain explicit with counts. Gate on preserving native behavior, not an
  arbitrary absolute accuracy score or beating seasonal naive on every task.
- Use `--profile performance` for optimization comparisons. Match native and
  vLLM HTTP boundaries, payloads, pins, precision, context, horizon, batch size,
  and client concurrency. Exclude startup, warmups, and numerical checks from
  timed samples. Under concurrency, throughput is completions / elapsed wall
  time. Retain failures and partial samples; failed points are not speedups.
- Save raw samples, error scores, full commit/dirty state, hardware, and cleanup
  evidence in ignored results directories. NVML sampled residency is not Torch
  allocator memory. Fewer than 100 requests is a harness smoke check; repeat and
  counterbalance backend order before interpreting performance differences.
- These profiles remain manual on the A100. A small Modal contract check does
  not load real models or replace model parity, accuracy, or performance checks.
  Do not add a noisy performance merge threshold or silently loosen parity
  tolerances to make an optimization pass.
- For [continuous batching](docs/continuous-batching.md), keep the pinned V1
  synchronous scheduler/worker hooks isolated. Prompt markers count forecast
  blocks, never observations. Count all active caches toward global numerical
  capacity; vLLM owns bounded HTTP intake and cancellation. Shared AR calls need
  per-row cache writes when positions differ and independently owned request
  buffers. Preserve native rotary positions and attention masks.
  Serving changes need real staggered-arrival, completion, error/cancellation,
  parity and peak-memory checks. Keep timing passes untraced and queue diagnostics
  separately labeled.
- Follow [numerical admission](docs/numerical-admission.md) for each model's limits.
  Request overload must fail before EngineCore submission. Split scheduled work
  within model-call budgets; validate the full resident slot envelope at startup.
  JSON overrides may tighten measured ceilings only. The shared resident-byte
  bound excludes model scratch/weights; raising ceilings needs fresh GPU peak
  measurements. Run `tests/test_model_admission.py` with
  `VTSFM_RUN_ADMISSION_ADAPTER=toto2` or `moirai2` in that adapter's environment;
  candidate policy metadata is not measured model capacity. Do not silently
  raise Moirai's eight-row or Toto's sixteen-row candidate caps. Compact public
  outputs must not retain larger native scratch allocations through views.
- Shared IO has no pending map; output metadata travels in the versioned result
  tensor. See [request lifecycle](docs/request-lifecycle.md). For lifecycle
  changes, run the opt-in TimesFM offline/HTTP failure and cancellation suite
  on the idle GPU. Keep its test-only fault hooks out of normal serving. V1
  worker model exceptions can be engine-fatal; distinguish clean restart from
  recovery within a healthy engine. Offline synchronous `LLM` calls are
  sequential; concurrent clients use the HTTP/async path.

Keep handoffs short: what changed, why, checks run, and remaining limitations.
These conventions take inspiration from [vLLM's agent guide](https://github.com/vllm-project/vllm/blob/main/AGENTS.md),
adapted to a small research team with lightweight local validation.
