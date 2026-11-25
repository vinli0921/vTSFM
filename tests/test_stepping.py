"""Block planning preserves identities and bounds padded cache work."""

import subprocess
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vtsfm.stepping import BlockRequest, plan_block_calls


def request(identity, rows=1, bucket=('decode', 8), work=10):
    return BlockRequest(identity, rows, bucket, work)


def plan(requests, rows=8, work=100, fusion=8):
    return plan_block_calls(requests, max_rows=rows, max_work=work, fusion_rows=fusion)


def test_compatible_nonadjacent_slots_share_a_call_without_changing_identity():
    assert plan([request('a', 2), request('new', bucket=('prefill', 32)), request('b', 3)]) == (
        ('a', 'b'), ('new',))


def test_mixed_cache_capacity_charges_padding_instead_of_summing_own_work():
    # Own work totals 3*2+1*50=56, but a shared padded call costs4*50=200.
    assert plan([request('short', 3, work=2), request('long', 1, work=50)], work=100) == (
        ('short',), ('long',))


def test_row_cap_and_standalone_fusion_fallback_preserve_every_request():
    assert plan([request('a', 3), request('b', 3), request('large', 7), request('c', 1)],
                rows=8, work=100, fusion=4) == (('a',), ('b',), ('large',), ('c',))


def test_completion_cancellation_and_reordering_do_not_retain_planner_state():
    assert plan([request('a'), request('b')]) == (('a', 'b'),)
    assert plan([request('b'), request('c')]) == (('b', 'c'),)
    assert plan([request('c')]) == (('c',),)
    assert plan([]) == ()


@pytest.mark.parametrize('requests', [
    [request('a'), request('a')], [request('large', 9)], [request('costly', work=101)],
])
def test_invalid_duplicate_or_oversized_requests_fail_before_execution(requests):
    with pytest.raises(ValueError):
        plan(requests)


def test_import_does_not_load_a_model_runtime():
    subprocess.run([sys.executable, '-c', 'import sys; import vtsfm.stepping; '
                    'assert not {"torch","vllm","timesfm"}&sys.modules.keys()'], check=True)


def test_direct_launch_requires_the_block_bridge_and_supported_execution_mode():
    from vtsfm.stepping import validate_block_configuration

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls='vtsfm.continuous_scheduler.ForecastBlockScheduler',
                                         async_scheduling=False, enable_chunked_prefill=True),
        parallel_config=SimpleNamespace(worker_cls='vtsfm.block_worker.ForecastBlockWorker',
                                        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=SimpleNamespace(enforce_eager=True, runner_type='pooling'),
        use_v2_model_runner=False, speculative_config=None)
    validate_block_configuration(config)
    config.parallel_config.worker_cls = 'auto'
    with pytest.raises(ValueError, match='ForecastBlockWorker'):
        validate_block_configuration(config)
    config.parallel_config.worker_cls = 'vtsfm.block_worker.ForecastBlockWorker'
    config.scheduler_config.scheduler_cls = None
    with pytest.raises(ValueError, match='ForecastBlockScheduler'):
        validate_block_configuration(config)


@dataclass
class _FakeState:
    identity: str
    blocks: int
    rows: int = 1
    blocks_done: int = 0
    values: list = field(default_factory=list)

    @property
    def bucket(self):
        return ('prefill', 0) if not self.blocks_done else ('decode', self.blocks_done)


class _FakeStepper:
    """Deterministic resumable adapter; values fingerprint request/row/block."""

    def advance(self, states):
        assert len({state.bucket for state in states}) == 1
        for state in states:
            state.values.append(tuple((state.identity, row, state.blocks_done) for row in range(state.rows)))
            state.blocks_done += 1

    prefill = advance

    def complete(self, state):
        return state.blocks_done == state.blocks

    def finalize(self, state):
        assert self.complete(state)
        return tuple(state.values)

    def release(self, state):
        state.values.clear()


def test_resumable_adapter_late_arrival_completion_reordering_and_cancellation():
    adapter = _FakeStepper()
    active = {'a': _FakeState('a', 4, 2), 'e': _FakeState('e', 4), 'short': _FakeState('short', 1)}
    results, shared = {}, []

    def iteration(order):
        requests = [BlockRequest(key, active[key].rows, active[key].bucket, 10) for key in order]
        for group in plan_block_calls(requests, max_rows=4, max_work=40, fusion_rows=4):
            states = [active[key] for key in group]
            shared.append(group)
            adapter.advance(states)
            for key, state in zip(group, states):
                if adapter.complete(state):
                    results[key] = adapter.finalize(state)
                    adapter.release(state)
                    del active[key]

    iteration(('a', 'e', 'short'))
    assert 'short' in results and active['a'].blocks_done == 1
    active['later'] = _FakeState('later', 2)
    iteration(('later', 'e', 'a'))  # New prefill and an existing shared decode.
    assert ('e', 'a') in shared and active['a'].blocks_done == 2
    cancelled = active.pop('e')
    adapter.release(cancelled)
    adapter.release(cancelled)  # Cleanup is safe even when repeated.
    assert cancelled.values == []
    while active:
        iteration(tuple(reversed(active)))
    assert 'e' not in results and set(results) == {'a', 'short', 'later'}
    assert results['a'] == tuple(tuple(('a', row, block) for row in range(2)) for block in range(4))
    assert results['later'] == ((('later', 0, 0),), (('later', 0, 1),))


def test_moirai_forecast_markers_match_native_64_sample_rounds(monkeypatch):
    from vtsfm.stepping import forecast_blocks

    monkeypatch.setenv('VTSFM_ADAPTER', 'moirai2')
    monkeypatch.setenv('VTSFM_EXPERIMENTAL_MOIRAI_BLOCKS', '1')
    assert [forecast_blocks(h) for h in (1, 64, 65, 96, 128, 129, 1024)] == [1, 1, 2, 2, 2, 3, 16]
