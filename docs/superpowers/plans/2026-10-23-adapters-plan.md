# TSFM Adapter Architecture Implementation Plan

> **For agentic workers:** Use scoped parallel implementation and independent final review.

**Goal:** Make Chronos-2 the first adapter of a general vLLM extension, with a separate shared batching planner.

**Architecture:** CPU-safe adapter descriptions select lazy model/IO implementations.
Shared batching operates only on metadata; adapter-specific numerical execution
remains in `adapters/chronos2/`. Keep existing entry points and serving behavior.

**Tech Stack:** Python dataclasses, standard library, existing vLLM/Chronos pins, pytest.

**Spec:** ../specs/2026-10-23-adapters-design.md

## Global constraints

Only Chronos-2 is supported. Preserve old imports, commands, model revision,
dependency pins, FP32/eager settings and benchmark methodology. No live batching,
new kernels, or GPU speedup claims. No shared VM service interruption.

## Review focus

- Discovery, CPU CLI and registry must not import Torch/vLLM/Chronos.
- Changing adapter metadata must affect registration, IO selection and launch.
- Requests must remain independent even when local group IDs would collide.
- Incompatible settings and row budgets must not silently change task semantics.
- Existing installed module paths, native launch and benchmark references must work.

## Tasks

- [x] Adapter integration (root): test descriptor/registry and CLI selection first;
  move implementation into `adapters/chronos2/`, preserve aliases, select all launch
  settings through metadata, add `adapters` CLI listing and run existing tests.
  Descriptor fields: name, model_id, model_revision, architectures, model_class,
  io_processor_class, upstream_io_plugin, native_server_module, vllm_args,
  supports_group_attention (default false, true for Chronos-2's inherited
  capability). Registry: list_adapters(), get_adapter(name).
- [x] Shared batching (scoped implementer): tests first for compatible grouping,
  mixed keys, budgets, globally unique group IDs, FIFO ordering, output routing,
  duplicate IDs, invalid metadata and oversized atomic requests. Implement the
  exact types/functions in the spec in `src/vtsfm/batching.py`; no GPU imports.
- [x] Documentation (scoped implementer): update README, architecture and
  contribution guide to explain boundaries, actual implemented features, adding
  an adapter, planning-only batching, first adapter limitations and future work.
- [x] Verify: full CPU suite, wheel, lazy CPU commands, independent code review,
  fresh install; GPU check only if an idle separate GPU is available.
- [x] Commit and publish a feature branch/PR without changing shared main.

## Implementation choices

The request supplies the architectural direction; conservative metadata-only
planning and compatibility shims are implementation choices to keep this change
reviewable and preserve the working baseline. Live execution integration is a
separate performance task.

## Verification record

150 CPU tests passed on Python 3.11.17. A built wheel installed into a fresh
environment passed adapter discovery, lazy entry-point loading, group planning,
output routing, and CLI dry-run checks without GPU packages. Independent review
found no important issues. Benchmark archives now capture all adapter/shared source.
GPU startup was not rerun: SSH to the existing instance timed out; no remote
service or environment was changed. `upstream` is a generic backend alias, while
`chronos2` remains accepted for backward compatibility.
Published as [draft PR #1](https://github.com/vinli0921/vTSFM/pull/1); the branch's
push and pull-request CPU checks both passed.
