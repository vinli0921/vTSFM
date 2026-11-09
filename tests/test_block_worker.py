"""Pinned serving-environment checks of block-slot identity and abort cleanup."""

import os
from types import SimpleNamespace

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_ROUTING_ENGINE') != '1',
    reason='Requires the pinned vLLM serving environment; no checkpoint needed')]


def _state(wire_id, *, horizon=1024, progress=0):
    import torch
    from vtsfm.adapters.transport import encode_metadata

    encoded = encode_metadata(wire_id, ['sensor'], [32], horizon, [0.5])
    field = SimpleNamespace(data=torch.tensor(list(encoded), dtype=torch.uint8))
    return SimpleNamespace(num_computed_tokens=progress,
        num_prompt_tokens=(horizon + 127) // 128,
        mm_features=[SimpleNamespace(modality='timeseries', data={'request_metadata': field})])


def test_block_slots_keep_worker_identity_order_and_cpu_progress():
    from vtsfm.block_worker import block_slots

    runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=2, req_ids=['internal-b', 'internal-a']),
        requests={'internal-a': _state('wire-a', progress=3),
                  'internal-b': _state('wire-b', horizon=129, progress=1)})
    slots = block_slots(runner)
    assert [(slot.request_id, slot.wire_id, slot.progress, slot.blocks) for slot in slots] == [
        ('internal-b', 'wire-b', 1, 2), ('internal-a', 'wire-a', 3, 8)]
    runner.input_batch.num_reqs = 0
    assert block_slots(runner) is None


@pytest.mark.parametrize('change', [
    {'num_prompt_tokens': 1}, {'num_computed_tokens': -1}, {'num_computed_tokens': 8},
    {'mm_features': []},
])
def test_invalid_block_slot_layout_is_rejected(change):
    from vtsfm.block_worker import block_slots

    state = _state('wire')
    for field, value in change.items():
        setattr(state, field, value)
    runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1, req_ids=['internal']),
                             requests={'internal': state})
    with pytest.raises(ValueError):
        block_slots(runner)


def test_finished_only_worker_step_forgets_before_upstream_processing(monkeypatch):
    from vtsfm.block_worker import ForecastBlockWorker
    from vtsfm.worker import ForecastWorker

    events = []
    executor = SimpleNamespace(forget=lambda ids: events.append(('forget', set(ids))))
    worker = object.__new__(ForecastBlockWorker)
    worker.model_runner = SimpleNamespace(get_model=lambda: SimpleNamespace(block_executor=executor))

    def upstream(self, scheduler_output):
        events.append(('upstream', scheduler_output.total_num_scheduled_tokens))
        return 'housekeeping-completed'

    monkeypatch.setattr(ForecastWorker, 'execute_model', upstream)
    result = worker.execute_model(SimpleNamespace(finished_req_ids={'cancelled'}, total_num_scheduled_tokens=0))
    assert result == 'housekeeping-completed'
    assert events == [('forget', {'cancelled'}), ('upstream', 0)]
