"""Exercise the full CLI flow without Torch, servers, GPU, or real HTTP."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]


@pytest.fixture
def simulation(monkeypatch, tmp_path):
    from benchmarks import continuous as module
    import benchmarks.run as runner
    from vtsfm import cli
    import requests
    events = []
    state = {'count': 0, 'reject_warmup': False}

    class Monitor:
        def __init__(self, *args):
            pass
        def metadata(self):
            return {'fake_gpu': True}
        def wait_idle(self):
            events.append('idle')
        def assert_idle(self):
            pass
        def close(self):
            events.append('monitor_closed')

    class Owner:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append('owned_cleanup')
        def spawn(self, *args):
            return object()
        def stop(self, *args):
            events.append('owned_stop')

    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            events.append(('sequential_timeout', kwargs['timeout']))
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {'data': {'predictions': []}})

    def replay(url, arrivals, **kwargs):
        warmup = kwargs['request_prefix'].startswith('warmup-')
        events.append(('replay', warmup, len(arrivals), kwargs.get('timeout', 120)))
        state['count'] += len(arrivals)
        rejected = int(warmup and state['reject_warmup'])
        result = {'offered': len(arrivals), 'completed': len(arrivals) - rejected,
                  'rejected': rejected, 'failed': 0, 'samples': [], 'nvml_samples': []}
        Path(kwargs['record_path']).write_text(json.dumps(result))
        return result

    def queue(url):
        events.append(('queue', state['count']))
        return {'count': state['count'], 'sum': float(state['count'])}

    def allocator(url, reset=False):
        events.append(('allocator_reset', reset))
        return {}

    def drain(url, path):
        events.append('drained')
        path.write_text(json.dumps({'status': 'passed'}))

    monkeypatch.setattr(runner, 'GpuMonitor', Monitor)
    monkeypatch.setattr(runner, 'OwnedProcesses', Owner)
    monkeypatch.setattr(runner, 'ensure_port_free', lambda *args: None)
    monkeypatch.setattr(runner, 'wait_ready', lambda *args, **kwargs: None)
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(cli, 'build_vllm_command', lambda **kwargs: (['fake-server'], {}))
    monkeypatch.setattr(module, 'replay', replay)
    monkeypatch.setattr(module, '_queue_counter', queue)
    monkeypatch.setattr(module, '_allocator_stats', allocator)
    monkeypatch.setattr(module, '_drain_requests', drain)
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda name: 'fake')
    monkeypatch.setattr(module, '_cases', lambda *args: [
        {'name': f'case{index}', 'data': {'inputs': [{'target': [1.]}]}} for index in range(4)])
    output = tmp_path / 'results'
    args = ['--data-dir', str(tmp_path), '--output', str(output), '--modes', 'continuous',
            '--scenarios', 'saturated', '--requests', '3', '--diagnostic-requests', '0']
    return module, args, output, events, state


@pytest.mark.parametrize('warmups', [0, 100])
def test_warmup_is_separate_and_precedes_primary_counter_and_memory_baselines(simulation, monkeypatch, warmups):
    module, args, output, events, state = simulation
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'compile')
    assert module.main(args + ['--warmup-requests', str(warmups), '--warmup-timeout', '600']) == 0
    primary = json.loads((output / 'continuous/saturated.json').read_text())
    assert primary['completed'] == 3 and primary['metrics_queue_count'] == 3
    assert [event for event in events if isinstance(event, tuple) and event[0] == 'queue'] == [
        ('queue', warmups), ('queue', warmups + 3)]
    assert ('sequential_timeout', 600.) in events
    if warmups:
        warm = json.loads((output / 'continuous/warmup-saturated.json').read_text())
        assert warm['completed'] == 100 and 'untimed' in warm['instrumentation']
        assert events.index(('replay', True, 100, 600.)) < events.index(('allocator_reset', True))
        assert events.index('drained') < events.index(('queue', 100))
    else:
        assert not (output / 'continuous/warmup-saturated.json').exists()
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['results']) == 1 and summary['results'][0]['completed'] == 3
    assert summary['settings'].startswith('compile FP32')
    assert 'owned_cleanup' in events and events[-1] == 'monitor_closed'


def test_rejected_warmup_retains_evidence_and_prevents_primary_measurement(simulation):
    module, args, output, events, state = simulation
    state['reject_warmup'] = True
    with pytest.raises(RuntimeError, match='warmup failed'):
        module.main(args + ['--warmup-requests', '5'])
    assert json.loads((output / 'continuous/warmup-saturated.json').read_text())['rejected'] == 1
    assert not (output / 'continuous/saturated.json').exists()
    assert not any(isinstance(event, tuple) and event[0] == 'queue' for event in events)
    assert 'owned_cleanup' in events and events[-1] == 'monitor_closed'


@pytest.mark.parametrize('flag,value', [('--warmup-requests', '-1'), ('--warmup-timeout', '0'),
                                       ('--warmup-timeout', 'nan'), ('--warmup-timeout', 'inf')])
def test_invalid_warmup_options_fail_before_data_or_gpu(simulation, flag, value):
    module, args, output, events, state = simulation
    with pytest.raises(SystemExit) as error:
        module.main(args + [flag, value])
    assert error.value.code == 2 and events == [] and not output.exists()


def load_benchmark():
    from benchmarks import continuous
    return continuous


def test_drain_observes_live_worker_even_after_engine_gauges_reach_zero(monkeypatch, tmp_path):
    import requests
    module = load_benchmark()
    snapshots = iter([{'worker_requests': 1, 'worker_slots': 1, 'state_count': 0, 'cache_bytes': 0},
                      {'worker_requests': 0, 'worker_slots': 0, 'state_count': 0, 'cache_bytes': 0}])
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            assert kwargs['json']['method'] == 'bench_request_state'
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': [next(snapshots)]})
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(module, '_queue_counter', lambda url: {'running': 0, 'waiting': 0})
    monkeypatch.setattr(module.time, 'sleep', lambda delay: None)
    path = tmp_path / 'drain.json'
    result = module._drain_requests('unused', path)
    assert result['status'] == 'passed' and len(result['observations']) == 2
    assert result['observations'][0]['worker']['worker_slots'] == 1
    assert json.loads(path.read_text())['status'] == 'passed'


def test_drain_never_imputes_missing_gauges_as_zero(monkeypatch, tmp_path):
    import requests
    module = load_benchmark()
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {'results': [{'worker_requests': 0, 'worker_slots': 0}]})
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(module, '_queue_counter', lambda url: {})
    path = tmp_path / 'missing.json'
    with pytest.raises(RuntimeError, match='gauges are unavailable'):
        module._drain_requests('unused', path)
    assert json.loads(path.read_text())['status'] == 'failed'


def test_queue_counter_preserves_running_waiting_gauges_and_queue_totals(monkeypatch):
    import requests
    module = load_benchmark()
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None, text='\n'.join([
                'vllm:num_requests_running{model_name="one"} 0.0',
                'vllm:num_requests_waiting{model_name="one"} 2.0',
                'vllm:request_queue_time_seconds_sum{model_name="one"} 3.5',
                'vllm:request_queue_time_seconds_count{model_name="one"} 10.0']))
    monkeypatch.setattr(requests, 'Session', Session)
    assert module._queue_counter('unused') == {'running': 0., 'waiting': 2., 'sum': 3.5, 'count': 10.}


def test_busy_drain_timeout_preserves_last_owned_state(monkeypatch, tmp_path):
    import requests
    module = load_benchmark()
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {'results': [{'worker_requests': 1, 'worker_slots': 1}]})
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(module, '_queue_counter', lambda url: {'running': 0, 'waiting': 0})
    path = tmp_path / 'busy.json'
    with pytest.raises(TimeoutError, match='did not drain'):
        module._drain_requests('unused', path, timeout=0)
    saved = json.loads(path.read_text())
    assert saved['status'] == 'failed' and saved['observations'][-1]['worker']['worker_slots'] == 1


@pytest.mark.parametrize('blocks', [False, True])
def test_worker_drain_snapshot_includes_native_slots_and_block_cache_ownership(monkeypatch, blocks):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    spec = importlib.util.spec_from_file_location('proposal_memory', CHECKOUT / 'src/vtsfm/benchmark_memory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = SimpleNamespace()
    if blocks:
        model.block_executor = SimpleNamespace(snapshot=lambda: {'state_count': 2, 'cache_bytes': 1024})
    worker = module.MemoryMetrics()
    worker.model_runner = SimpleNamespace(requests={'one': object()},
        input_batch=SimpleNamespace(num_reqs=1), get_model=lambda: model)
    state = worker.bench_request_state()
    assert state['worker_requests'] == state['worker_slots'] == 1
    assert (state.get('cache_bytes'), state.get('state_count')) == ((1024, 2) if blocks else (None, None))
