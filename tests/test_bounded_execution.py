"""Checkpoint-free tests of the live whole-forecast planner on CPU tensors."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip('torch')


def _request(torch, identity, rows=2, horizon=129, lengths=None):
    lengths = lengths or (32,) * rows
    context = torch.full((rows, max(lengths)), float('nan'))
    for index, length in enumerate(lengths):
        context[index, -length:] = float(ord(identity[0]) + index)
    return SimpleNamespace(request_id=identity, item_ids=tuple(f'{identity}-{i}' for i in range(rows)),
                           context=context, context_lengths=lengths, horizon=horizon)


class _Runtime:
    """Distinct row fingerprints make incorrect splitting/routing observable."""

    def __init__(self, torch):
        self.torch, self.calls = torch, []

    def _values(self, context, horizon):
        return context[:, -1, None, None].expand(len(context), 9, horizon).clone()

    def forecast(self, context, horizon):
        self.calls.append(('single', len(context), horizon))
        return self._values(context, horizon)

    def execute(self, batch):
        from vtsfm.execution import ExecutionResult

        context = self.torch.cat([request.context for request in batch.requests])
        self.calls.append(('batch', len(context), batch.plan.key.prediction_length))
        return ExecutionResult(batch, self._values(context, batch.plan.key.prediction_length))


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_live_planner_splits_individually_valid_requests_at_aggregate_work_limit(torch, adapter, tmp_path):
    from vtsfm.adapters import get_adapter
    from vtsfm.adapters.execution_batch import execute_scheduled
    from vtsfm.admission import resolve_cost

    spec = get_adapter(adapter)
    cost = resolve_cost(spec)
    limits = replace(spec.admission_limits, max_execution_work=cost(4, 32, 129).work_units)
    requests = [_request(torch, identity) for identity in ('A', 'B', 'C')]
    for request in requests:
        limits.validate_request(cost(2, 32, 129))
    runtime = _Runtime(torch)
    trace = tmp_path / 'calls.jsonl'
    outputs = list(execute_scheduled(runtime, spec, requests, 8, str(trace),
                                    limits=limits, cost_for_shape=cost))
    assert runtime.calls == [('batch', 4, 129), ('single', 2, 129)]
    assert len(outputs) == len(requests)
    for request, result in zip(requests, outputs, strict=True):
        assert torch.equal(result, runtime._values(request.context, request.horizon))
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event['request_ids'] for event in events] == [['A', 'B'], ['C']]
    assert all(event['execution_work'] <= limits.max_execution_work for event in events)


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_live_planner_keeps_ragged_incompatible_and_above_fusion_cap_requests_isolated(torch, adapter):
    from vtsfm.adapters import get_adapter
    from vtsfm.adapters.execution_batch import execute_scheduled
    from vtsfm.admission import resolve_cost

    spec = get_adapter(adapter)
    cost = resolve_cost(spec)
    requests = [_request(torch, 'A', rows=3),  # Fits admission, exceeds opt-in fusion cap.
                _request(torch, 'B', lengths=(31, 32)),  # Ragged remains isolated.
                _request(torch, 'C', horizon=128),
                _request(torch, 'D', horizon=129)]  # Different horizon cannot fuse with C.
    runtime = _Runtime(torch)
    outputs = list(execute_scheduled(runtime, spec, requests, 2,
                                    limits=spec.admission_limits, cost_for_shape=cost))
    assert runtime.calls == [('single', 3, 129), ('single', 2, 129),
                             ('single', 2, 128), ('single', 2, 129)]
    for request, result in zip(requests, outputs, strict=True):
        assert torch.equal(result, runtime._values(request.context, request.horizon))
