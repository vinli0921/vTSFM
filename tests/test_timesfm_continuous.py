"""Opt-in real TimesFM HTTP evidence for forecast-block continuous batching.

    CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_CONTINUOUS=1 PYTHONPATH=src:tests:. \\
      .venv-timesfm25/bin/python -m pytest tests/test_timesfm_continuous.py -q -s

Deterministic synthetic inputs are correctness probes, not accuracy workloads.
Only the test worker installs gates; normal serving has no sleeps or gate files.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time

import pytest

import test_routing_engine
from test_timesfm_http_batching import _case, _compare, _post, _run_wave, _wait_for_submission
from test_timesfm_lifecycle import _disconnectable, _request


ROOT = Path(__file__).resolve().parents[1]
gpu = test_routing_engine.gpu
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_CONTINUOUS') != '1',
    reason='Set VTSFM_RUN_TIMESFM_CONTINUOUS=1 on an idle Linux GPU')]


def _wait(predicate, message, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            pytest.fail(message)
        time.sleep(.02)


def _events(path):
    events = []
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live append can be observed between write and newline.
        for key in ('request_ids', 'finished_ids'):
            if key in event:
                event[key] = [identity.rsplit(':', 1)[0] for identity in event[key]]
        events.append(event)
    return events


def _rpc(server, method, *args):
    response = server['control'].post(server['url'] + '/collective_rpc', json={
        'method': method, 'args': list(args), 'timeout': 20}, timeout=25)
    response.raise_for_status()
    result = response.json()['results']
    assert len(result) == 1
    return result[0]


def _empty(server):
    def check():
        state = _rpc(server, 'continuous_snapshot')
        if any(state[key] for key in ('state_count', 'active_rows', 'pending_outputs',
                                      'worker_requests', 'worker_slots')):
            return None
        assert state['cache_bytes'] == 0
        return state
    return _wait(check, 'Continuous state did not return to its empty baseline')


@contextmanager
def _server(directory, monitor, *, continuous=True, slots=3, queue_limit=16, fusion_rows=8):
    import requests
    from benchmarks.run import OwnedProcesses, ensure_port_free, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command

    directory.mkdir(exist_ok=True)
    monitor.wait_idle()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ensure_port_free('127.0.0.1', port)
    command, environment = build_vllm_command(
        adapter='timesfm25', port=port, benchmark=True, max_num_seqs=slots,
        forecast_batch_rows=fusion_rows if slots > 1 else None,
        continuous_batching=continuous, forecast_queue_limit=queue_limit)
    if continuous:
        command[command.index('--worker-extension-cls') + 1] = 'continuous_probe.ContinuousProbeExtension'
    command += ['--gpu-memory-utilization', '.35', '--mm-processor-cache-gb', '0', '--enable-log-requests']
    trace = directory / 'steps.jsonl'
    environment.update(VTSFM_CONTINUOUS_TRACE=str(trace), OMP_NUM_THREADS='4', NVIDIA_TF32_OVERRIDE='0',
                       PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]))
    url = f'http://127.0.0.1:{port}'
    with OwnedProcesses(directory) as owner, requests.Session() as control:
        control.trust_env = False
        child = owner.spawn(command, environment, 'continuous-server')
        wait_ready(child, url, control, timeout=180)
        server = {'url': url, 'control': control, 'child': child, 'trace': trace,
                  'model': get_adapter('timesfm25').model_id,
                  'log': directory / 'continuous-server.log', 'gate': directory / 'gate'}
        if continuous:
            _rpc(server, 'continuous_install', str(server['gate']))
        yield server
    monitor.wait_idle()


def _arm(server, kind):
    for suffix in ('armed', 'entered', 'release'):
        (server['gate'] / f'{kind}-{suffix}').unlink(missing_ok=True)
    (server['gate'] / f'{kind}-armed').write_text('armed')


def _release(server):
    for kind in ('prefill', 'decode'):
        (server['gate'] / f'{kind}-release').write_text('released')


def _identity(label):
    return f'pooling-{label}'


def test_real_timesfm_staggered_join_independent_completion_and_parity(gpu, tmp_path):
    cases = {
        'a': _case('a', (128,), -2., horizon=1024),
        'e': _case('e', (128,), 3., horizon=1024, quantiles=(.2, .8)),
        'c': _case('c', (128,), -1., horizon=128),
        'b': _case('b', (128,), 1., horizon=128),
        'd': _case('d', (128,), -.5, horizon=256),
        'patch31': _case('patch31', (31,), -1., horizon=127),
        'patch33': _case('patch33', (33,), 1., horizon=129),
        'ragged': _case('ragged', (31, 32, 33), -1., horizon=513),
    }
    with _server(tmp_path / 'reference', gpu, continuous=False, slots=1) as server:
        expected = {label: _post(server['url'], server['model'], f'reference-{label}', data)
                    for label, data in cases.items()}
    snapshots = []
    with _server(tmp_path / 'continuous', gpu) as server:
        for label in ('patch31', 'patch33', 'ragged', 'a', 'patch31'):
            _compare(_post(server['url'], server['model'], f'isolated-{label}', cases[label]),
                     expected[label], cases[label])
        snapshots.append(_empty(server))
        _arm(server, 'prefill')
        _arm(server, 'decode')
        control, url = server['control'], server['url']
        with ThreadPoolExecutor(max_workers=5) as clients:
            control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
            pending = {}
            try:
                for label in ('a', 'e', 'c'):
                    pending[label] = clients.submit(_post, url, server['model'], f'join-{label}', cases[label])
                    _wait_for_submission(server['log'], f'join-{label}', pending[label])
                control.post(url + '/resume', timeout=10).raise_for_status()
                _wait(lambda: (server['gate'] / 'prefill-entered').exists(), 'Prefill gate was not reached')
                for label in ('b', 'd'):
                    pending[label] = clients.submit(_post, url, server['model'], f'join-{label}', cases[label])
                    _wait_for_submission(server['log'], f'join-{label}', pending[label])
                (server['gate'] / 'prefill-release').write_text('later requests submitted')
                _wait(lambda: (server['gate'] / 'decode-entered').exists(), 'Shared decode gate was not reached')
                _compare(pending['c'].result(timeout=10), expected['c'], cases['c'])
                assert not pending['a'].done() and not pending['e'].done(), 'Long requests completed behind gate'
                (server['gate'] / 'decode-release').write_text('short response completed independently')
                for label, future in pending.items():
                    _compare(future.result(timeout=120), expected[label], cases[label])
            finally:
                _release(server)
                control.post(url + '/resume', timeout=10).raise_for_status()
        snapshots.append(_empty(server))
        events = _events(server['trace'])
        steps = [event for event in events if event.get('event') == 'step']
        starts = {label: next(event['timestamp_ns'] for event in steps
                             if _identity(f'join-{label}') in event['request_ids']) for label in ('a', 'e', 'c', 'b', 'd')}
        ends = {label: next(event.get('end_ns', event['timestamp_ns']) for event in steps
                           if _identity(f'join-{label}') in event.get('finished_ids', []))
                for label in ('a', 'e', 'c', 'b', 'd')}
        assert starts['a'] < starts['b'] < ends['a']
        assert ends['c'] <= starts['b']
        assert ends['b'] <= starts['d'] < ends['a']
        shared = [event for event in steps if event.get('stage') == 'decode'
                  and {_identity('join-a'), _identity('join-e')} <= set(event['request_ids'])]
        assert shared, 'No actual AR model call batched active requests together'
        mixed = [event for event in steps if event.get('stage') == 'decode'
                 and len(set(event.get('cache_positions', []))) > 1]
        assert mixed, 'Staggered arrivals never shared a decode call across cache positions'
        assert all(event['cache_position'] is None for event in mixed)
        assert all(event['rows'] <= 8 for event in steps)
        assert server['child'].poll() is None
    evidence = {'continuous_timesfm': 'passed', 'synthetic': True,
                'starts_ns': starts, 'ends_ns': ends, 'shared_decode_calls': shared,
                'mixed_position_decode_calls': mixed, 'empty_state_snapshots': snapshots}
    (tmp_path / 'continuous-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'continuous_timesfm': 'passed', 'shared_decode_calls': len(shared)}))


def test_real_timesfm_queue_overload_and_cancel_release_only_owned_state(gpu, tmp_path):
    case = _case('healthy', (128,), -1., horizon=1024)
    small = _case('small', (128,), 2., horizon=128)
    snapshots = []
    with _server(tmp_path / 'lifecycle', gpu, slots=2, queue_limit=3) as server:
        expected = _post(server['url'], server['model'], 'healthy-reference', case)
        for cycle in range(3):
            _arm(server, 'prefill')
            _arm(server, 'decode')
            connection = _disconnectable(server['url'], server['model'], f'cancel-{cycle}', case)
            try:
                _wait(lambda: (server['gate'] / 'prefill-entered').exists(), 'Cancel prefill gate was not reached')
                with ThreadPoolExecutor(max_workers=2) as clients:
                    neighbor = clients.submit(_post, server['url'], server['model'], f'neighbor-{cycle}', case)
                    queued = clients.submit(_post, server['url'], server['model'], f'queued-{cycle}', small)
                    _wait_for_submission(server['log'], f'neighbor-{cycle}', neighbor)
                    _wait_for_submission(server['log'], f'queued-{cycle}', queued)
                    response = server['control'].post(server['url'] + '/pooling',
                        json={'model': server['model'], 'task': 'plugin', 'data': small}, timeout=15)
                    assert response.status_code == 503, response.text
                    connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                    connection = None
                    _release(server)
                    _compare(neighbor.result(timeout=120), expected, case)
                    queued.result(timeout=120)
            finally:
                _release(server)
                if connection is not None:
                    connection.close()
            _compare(_post(server['url'], server['model'], f'healthy-after-{cycle}', case), expected, case)
            snapshots.append(_empty(server))

        # Every observation is a finite FP32 value. The unbiased standard
        # deviation exceeds FP32's range, so outer denormalization produces a
        # nonfinite forecast. This exercises an actual postprocessing failure,
        # with GPU request state already retired, alongside a healthy request.
        overflow = {'inputs': [{'item_id': 'finite-overflow',
                               'target': [3.4e38 if index % 2 == 0 else -3.4e38
                                          for index in range(128)]}],
                    'parameters': {'prediction_length': 129, 'quantile_levels': [.1, .5, .9]}}
        server['control'].post(server['url'] + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
        with ThreadPoolExecutor(max_workers=2) as clients:
            try:
                failed = clients.submit(_request, server['url'], server['model'], 'postprocess-overflow', overflow)
                healthy = clients.submit(_post, server['url'], server['model'], 'postprocess-healthy', case)
                _wait_for_submission(server['log'], 'postprocess-overflow', failed)
                _wait_for_submission(server['log'], 'postprocess-healthy', healthy)
            finally:
                server['control'].post(server['url'] + '/resume', timeout=10).raise_for_status()
            response = failed.result(timeout=120)
            assert response.status_code in (400, 500), response.text
            assert 'nonfinite' in response.text.lower(), response.text
            _compare(healthy.result(timeout=120), expected, case)
        postprocess_failure = {'http_status': response.status_code, 'error': response.text,
                               'healthy_peer_succeeded': True}
        snapshots.append(_empty(server))
        _compare(_post(server['url'], server['model'], 'healthy-after-postprocess-error', case), expected, case)
        snapshots.append(_empty(server))
        events = _events(server['trace'])
        cancels = [event for event in events if event.get('event') == 'release' and event.get('reason') == 'cancel']
        assert all(any(_identity(f'cancel-{cycle}') in event['request_ids'] for event in cancels)
                   for cycle in range(3))
        assert server['child'].poll() is None
    evidence = {'continuous_lifecycle': 'passed', 'cycles': 3, 'cancel_events': cancels,
                'postprocess_failure': postprocess_failure,
                'empty_state_snapshots': snapshots, 'synthetic': True}
    (tmp_path / 'continuous-lifecycle-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


def test_real_timesfm_persistent_cache_envelope_and_capacity_deferral(gpu, tmp_path):
    """Measure retained two-pass KV state as well as transient model scratch."""
    from dataclasses import asdict
    import importlib.metadata
    from benchmarks.run import Sampler
    from vtsfm.adapters import get_adapter
    from vtsfm.adapters.timesfm25.admission import DEFAULT_LIMITS

    limits = DEFAULT_LIMITS
    # Pinned architecture: two flip passes, K/V, 20 layers, 16 heads of dim80,
    # FP32. At most 16 active rows and 284 cache positions per row are admitted.
    cache_positions = (limits.max_context + 31) // 32 + 4 * ((limits.max_horizon - 1) // 128)
    cache_byte_bound = 2 * 2 * 20 * 16 * 80 * 4 * limits.max_execution_rows * cache_positions
    assert cache_byte_bound == 1_861_222_400
    four = _case('memory-four', (limits.max_context,) * 4, -1., horizon=limits.max_horizon)
    sixteen = {'inputs': [
        {'item_id': f'memory-sixteen-{row}', 'target': four['inputs'][row % 4]['target']}
        for row in range(16)], 'parameters': dict(four['parameters'])}
    small = _case('small', (128,), -1., horizon=128)
    profiles, states = [], []
    selected = get_adapter('timesfm25')

    with _server(tmp_path / 'memory', gpu, slots=4, fusion_rows=16) as server:
        _post(server['url'], server['model'], 'memory-warmup', small)
        expected = _post(server['url'], server['model'], 'memory-isolated-four', four)
        expected_sixteen = [dict(expected[row % 4], item_id=item['item_id'])
                            for row, item in enumerate(sixteen['inputs'])]
        states.append(_empty(server))

        for name, case, baseline in (('four-shared', four, expected),
                                     ('sixteen-deferred', sixteen, expected_sixteen)):
            _rpc(server, 'bench_memory_empty_cache')
            before = _rpc(server, 'bench_memory_reset')
            with Sampler(gpu, interval=.02) as sampler:
                completed = _run_wave(server['url'], server['control'], server['model'],
                    server['log'], name, ('request',) * 4, {'request': case})
                after = _rpc(server, 'bench_memory_stats')
                empty = _empty(server)
            for _, _, predictions in completed:
                _compare(predictions, baseline, case)
            states.append(empty)
            identities = {identity for identity, _, _ in completed}
            steps = [event for event in _events(server['trace'])
                     if event.get('event') == 'step' and identities & set(event['request_ids'])]
            assert steps and all(event['active_rows'] <= limits.max_execution_rows for event in steps)
            assert all(event['cache_bytes'] <= cache_byte_bound for event in steps)
            if name == 'four-shared':
                shared = [event for event in steps if event['stage'] == 'decode'
                          and set(event['request_ids']) == identities]
                assert shared and all(event['rows'] == 16 for event in shared)
                assert max(event['cache_bytes'] for event in steps) == cache_byte_bound
            else:
                # Each valid sixteen-row request consumes all resident capacity;
                # the other three wait instead of being staged/advanced together.
                assert all(len(event['request_ids']) == 1 and event['rows'] == 16 for event in steps)
            assert after['peak_allocated_bytes'] >= before['allocated_bytes'] > 0
            assert after['peak_reserved_bytes'] >= after['peak_allocated_bytes']
            profiles.append({'name': name, 'series_per_request': len(case['inputs']),
                             'concurrent_requests': 4, 'context': limits.max_context,
                             'horizon': limits.max_horizon, 'baseline': before, 'worker': after,
                             'peak_cache_bytes': max(event['cache_bytes'] for event in steps),
                             'peak_active_rows': max(event['active_rows'] for event in steps),
                             'peak_nvml_used_bytes': max(sample[1] for sample in sampler.samples),
                             'nvml_samples': sampler.samples, 'steps': steps})
        assert server['child'].poll() is None
    evidence = {'continuous_memory_envelope': 'passed', 'synthetic': True,
                'model': selected.model_id, 'revision': selected.model_revision,
                'gpu': gpu.metadata(), 'limits': asdict(limits),
                'packages': {name: importlib.metadata.version(name) for name in ('torch', 'vllm', 'timesfm')},
                'precision': 'FP32 eager, TF32 disabled', 'persistent_cache_byte_bound': cache_byte_bound,
                'memory_method': 'Torch peaks reset after empty_cache outside each probe; 20 ms NVML',
                'notes': ['Finite shape probes, not an exact total-VRAM guarantee.',
                          'Sixteen-row payload repeats the four independent synthetic histories.',
                          'Persistent cache count excludes weights, running stats, output buffers and transient scratch.'],
                'profiles': profiles, 'empty_state_snapshots': states}
    (tmp_path / 'continuous-memory-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'continuous_memory_envelope': 'passed',
        'profiles': [{'name': profile['name'], 'peak_cache_bytes': profile['peak_cache_bytes'],
                      'peak_allocated_bytes': profile['worker']['peak_allocated_bytes'],
                      'peak_reserved_bytes': profile['worker']['peak_reserved_bytes'],
                      'peak_nvml_used_bytes': profile['peak_nvml_used_bytes']} for profile in profiles]}))
