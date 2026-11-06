"""Execute compatible whole forecasts already scheduled by a serving runner."""

from itertools import groupby
import json
from pathlib import Path

import torch

from vtsfm.batching import BatchKey, BatchRequest, plan_batches
from vtsfm.execution import PreparedBatch, PreparedRequest


def execute_scheduled(runtime, spec, requests, max_rows, trace_path=None):
    """Yield [series, native quantiles, horizon] tensors in scheduler slot order."""
    capabilities = spec.execution_capabilities

    def eligible(request):
        width = request.context.shape[1]
        return (len(request.item_ids) <= max_rows
                and all(length == width for length in request.context_lengths))

    def trace(kind, members):
        if trace_path is None:
            return
        first = members[0]
        with Path(trace_path).open('a') as stream:
            stream.write(json.dumps({
                'kind': kind, 'request_ids': [request.request_id for request in members],
                'rows': sum(len(request.item_ids) for request in members),
                'horizon': first.horizon, 'context_length': first.context.shape[1],
            }) + '\n')

    def single(request):
        values = runtime.forecast(request.context, request.horizon)
        trace('single', (request,))
        return values

    # An incompatible request ends the current run. Oversized or ragged inputs
    # use isolated execution to preserve the existing HTTP contract.
    for can_batch, run in groupby(requests, key=eligible):
        if not can_batch:
            for request in run:
                yield single(request)
            continue

        by_id = {}
        metadata = []
        for request in run:
            by_id[request.request_id] = request
            # The runtime computes all native quantiles. HTTP selects each
            # request's quantiles after routing, so that projection can differ.
            key = BatchKey(spec.name, spec.model_id, spec.model_revision,
                           request.context.shape[1], request.horizon, capabilities.quantiles,
                           execution_options=capabilities.execution_options)
            metadata.append(BatchRequest(request.request_id, key, (1,) * len(request.item_ids)))

        for plan in plan_batches(metadata, max_rows=max_rows, max_requests=4):
            members = tuple(by_id[part.request_id] for part in plan.request_slices)
            if len(members) == 1:
                yield single(members[0])
                continue
            prepared = tuple(PreparedRequest(
                BatchRequest(request.request_id, plan.key, (1,) * len(request.item_ids)),
                request.context, request.context_lengths,
                torch.ones_like(request.context, dtype=torch.bool), request.item_ids, capabilities,
            ) for request in members)
            outputs = runtime.execute(PreparedBatch(plan, prepared)).split()
            trace('batch', members)
            for request in members:
                yield outputs[request.request_id].values
