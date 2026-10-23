# General TSFM extension with a first Chronos-2 adapter

The user requested a general TSFM extension for vLLM, with Chronos-2 as the first
adapter and shared batching separated from model-specific optimizations. This
refactor implements that boundary while preserving the verified serving behavior.

## Boundaries

- `adapters/base.py` describes an adapter using immutable, CPU-safe metadata.
  `adapters/registry.py` lists and resolves available adapters without importing
  Torch, vLLM, or model packages. Chronos-2 is the only implemented adapter.
- `adapters/chronos2/` owns its pinned model/integration, numerical IO, model
  registration target, native comparator, serving arguments, and future attention
  optimizations. Existing model/IO/native module paths remain compatibility shims.
- `plugin.py` registers adapter model architectures lazily and returns the IO
  class selected by `VTSFM_ADAPTER` (default `chronos2`). The CLI sets that variable
  explicitly. Both existing `vtsfm` entry-point names remain unchanged.
- CLI `serve --adapter chronos2` selects the model family independently of backend
  (`vtsfm`, upstream `chronos2`, or `native`). Unknown adapter names fail before
  launching a server. Existing commands keep the same defaults and GPU settings.
- `batching.py` is an independent metadata planner: conservative compatibility,
  row/request budgets, unique group IDs across requests, and row-aligned output
  routing. It does not import adapters or model runtimes and does not perform
  tensor packing, queueing, GPU execution, or change the live serving path.

## Shared batching contract

`BatchKey(adapter, model_id, model_revision, context_length, prediction_length,
quantile_levels, dtype='float32')` identifies compatible prepared work. Adapter
identifies the model family; checkpoint ID, revision and execution settings
must match. Different lengths/horizons/quantiles are conservatively separate.
An empty quantile tuple is valid for point-only outputs; numerical model
capabilities are validated by the adapter, not inferred by the planner.

`BatchRequest(request_id, key, group_sizes)` describes positive prepared row counts
for independent attention groups. The adapter is responsible for forming those
groups, including cross-learning semantics and any covariate rows. Requests are
atomic: no group or request is split to fit a budget.

`plan_batches(requests, *, max_rows, max_requests)` returns FIFO contiguous plans,
flushing on an incompatible key or a budget boundary. It rejects duplicate request
IDs, invalid shapes/settings, and individually oversized requests. Each plan has
`key`, `group_ids`, `request_slices` and `split_outputs(outputs)`, which routes
row-aligned results only. Model-specific target/covariate filtering stays in IO.

This is a CPU-testable reference for later scheduler/worker integration, not an
enabled cross-request batcher or a performance optimization claim.

## Preservation and verification

Keep exact model/revision/dependency pins, eager FP32 execution, prefix cache off,
`max_num_seqs=1`, upstream comparator, and existing benchmark results intact.
Preserve CPU-safe imports and shell onboarding. Test lazy registry/discovery,
adapter selection, conservative group planning, routing, and existing behavior.
Build a wheel and run the existing CPU suite. GPU verification must not interrupt
the shared VM's existing service; if occupied, report that limitation explicitly.

## Deferred work

Another model adapter, common public numerical HTTP schema, live queue admission,
multi-prompt vLLM tensor transport, actual tensor packing, grouped attention
kernels, concurrent/grouped workload benchmarks, and AssetOpsBench MCP integration.
