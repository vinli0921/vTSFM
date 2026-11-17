"""Manual real-TimesFM V1 offline, failure, and cancellation checks.

    CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_LIFECYCLE=1 PYTHONPATH=src:tests:. \\
      .venv-timesfm25/bin/python -m pytest tests/test_timesfm_lifecycle.py -q -s

Inputs are deterministic synthetic correctness probes, not accuracy or performance
workloads. Test-only instrumentation injects faults and gates a genuine TimesFM
forward. Normal serving does not load these hooks or expose their routes.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

import pytest

import test_routing_engine
from test_timesfm_http_batching import _case, _compare, _post, _wait_for_submission


gpu = test_routing_engine.gpu
ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get('VTSFM_RUN_TIMESFM_LIFECYCLE') != '1',
                       reason='Set VTSFM_RUN_TIMESFM_LIFECYCLE=1 on an idle Linux GPU'),
]


def _environment():
    environment = dict(os.environ)
    environment.update(VLLM_PLUGINS='vtsfm', VTSFM_ADAPTER='timesfm25',
                       VLLM_USE_V2_MODEL_RUNNER='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                       VLLM_KEEP_ALIVE_ON_ENGINE_DEATH='0',
                       NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4',
                       PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]))
    return environment


def _wait(predicate, message, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            pytest.fail(message)
        time.sleep(0.025)


def _rpc(control, url, method, *args):
    response = control.post(url + '/collective_rpc', json={
        'method': method, 'args': list(args), 'timeout': 20}, timeout=25)
    response.raise_for_status()
    results = response.json()['results']
    assert len(results) == 1
    return results[0]


def _api_state(control, url):
    response = control.get(url + '/lifecycle-test/state', timeout=10)
    response.raise_for_status()
    return response.json()


def _worker_empty(state):
    return not any(state[field] for field in (
        'worker_cached_requests', 'worker_slots', 'worker_encoder_cache',
        'worker_pooling_params', 'worker_pooling_states',
        'worker_has_execute_model_state', 'pooler_has_routes'))


def _quiescent(control, url):
    def check():
        api = _api_state(control, url)
        worker = _rpc(control, url, 'lifecycle_snapshot')
        if (api['api_unfinished'] or api['io_pending'] or api['live_preprocessed_contexts']
                or api['live_preprocessed_metadata']
                or not _worker_empty(worker)):
            return None
        return {**api, **worker}
    return _wait(check, 'Request state did not return to its empty baseline')


@contextmanager
def _server(directory, monitor, timesfm_forecast_mode=None):
    import requests
    from benchmarks.run import OwnedProcesses, ensure_port_free, wait_ready
    from vtsfm.cli import build_vllm_command

    directory.mkdir(exist_ok=True)
    monitor.wait_idle()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ensure_port_free('127.0.0.1', port)
    url = f'http://127.0.0.1:{port}'
    options = ({'timesfm_forecast_mode': timesfm_forecast_mode} if timesfm_forecast_mode
               else {'max_num_seqs': 4, 'forecast_batch_rows': 8})
    command, environment = build_vllm_command(adapter='timesfm25', port=port, benchmark=True, **options)
    # Import instrumentation only in this test's API process and worker.
    command = [sys.executable, '-c',
               'import lifecycle_probe; from vllm.entrypoints.cli.main import main; main()',
               *command[1:]]
    extension = 'GraphLifecycleProbeExtension' if timesfm_forecast_mode else 'LifecycleProbeExtension'
    command[command.index('--worker-extension-cls') + 1] = f'lifecycle_probe.{extension}'
    command += ['--gpu-memory-utilization', '0.35', '--mm-processor-cache-gb', '0',
                '--enable-log-requests']
    environment.update(_environment(), VLLM_SERVER_DEV_MODE='1')
    with OwnedProcesses(directory) as owner, requests.Session() as control:
        control.trust_env = False
        child = owner.spawn(command, environment, 'timesfm-lifecycle-server')
        wait_ready(child, url, control, timeout=360 if timesfm_forecast_mode else 180)
        _rpc(control, url, 'lifecycle_install', str(directory / 'gate'))
        yield url, control, directory / 'timesfm-lifecycle-server.log', directory / 'gate', child
    monitor.wait_idle()


def _request(url, model, request_id, data, timeout=30):
    import requests
    with requests.Session() as session:
        session.trust_env = False
        return session.post(url + '/pooling', headers={'X-Request-Id': request_id},
                            json={'model': model, 'task': 'plugin', 'data': data}, timeout=timeout)


def _disconnectable(url, model, request_id, data):
    """Send a complete HTTP request and keep its socket for explicit disconnect."""
    from urllib.parse import urlsplit
    address = urlsplit(url)
    body = json.dumps({'model': model, 'task': 'plugin', 'data': data}).encode()
    connection = socket.create_connection((address.hostname, address.port), timeout=10)
    header = (f'POST /pooling HTTP/1.1\r\nHost: {address.netloc}\r\n'
              f'X-Request-Id: {request_id}\r\nContent-Type: application/json\r\n'
              f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode()
    connection.sendall(header + body)
    return connection


def _wait_log(log, marker):
    _wait(lambda: marker in log.read_text(errors='replace'), f'Missing {marker!r} in {log}')


def _offline(directory, cases, monkeypatch):
    from vllm import LLM
    from vtsfm.adapters import get_adapter
    import lifecycle_probe

    for key, value in _environment().items():
        if key.startswith(('VLLM_', 'VTSFM_')):
            monkeypatch.setenv(key, value)
    selected = get_adapter('timesfm25')
    llm = LLM(
        model=selected.model_id, revision=selected.model_revision, config_format='vtsfm',
        runner='pooling', hf_overrides={'architectures': list(selected.architectures)},
        io_processor_plugin='vtsfm', dtype='float32', skip_tokenizer_init=True,
        enforce_eager=True, enable_prefix_caching=False, enable_mm_embeds=True,
        mm_processor_cache_gb=0, max_model_len=8192, max_num_batched_tokens=8192,
        max_num_seqs=4, gpu_memory_utilization=0.35,
        worker_cls='vtsfm.worker.ForecastWorker',
        worker_extension_cls='lifecycle_probe.LifecycleProbeExtension')
    try:
        llm.collective_rpc('lifecycle_install', args=(str(directory / 'offline-gate'),))

        def encode(data, engine=llm):
            outputs = engine.encode({'data': data}, pooling_task='plugin', use_tqdm=False)
            assert len(outputs) == 1 and outputs[0].finished
            return outputs[0].outputs['predictions']

        isolated = {label: encode(data) for label, data in cases.items()}
        snapshots = []
        for _ in range(3):
            for label in reversed(cases):
                _compare(encode(cases[label]), isolated[label], cases[label])
            for trigger in ('__fail_pre__', '__fail_submit__', '__fail_post__'):
                with pytest.raises((ValueError, RuntimeError)) as error:
                    encode(_case(trigger, (32,), -1.0))
                assert str(error.value), trigger
                # ExceptionInfo intentionally retains the traceback and its input
                # tensors; release this test-owned reference before checking IO.
                del error
                _compare(encode(cases['a']), isolated['a'], cases['a'])
            assert llm.llm_engine.get_num_unfinished_requests() == 0
            state = lifecycle_probe.io_snapshot()
            assert (state['io_pending'] == state['live_preprocessed_contexts']
                    == state['live_preprocessed_metadata'] == 0)

            def worker_quiescent(engine=llm):
                worker = engine.collective_rpc('lifecycle_snapshot')[0]
                return worker if _worker_empty(worker) else None

            worker = _wait(worker_quiescent, 'Offline worker retained completed request state')
            snapshots.append({**state, **worker})
        return isolated, snapshots
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


@pytest.mark.usefixtures('gpu')
def test_real_timesfm_offline_http_failure_and_cancellation(tmp_path, monkeypatch, gpu):
    from vtsfm.adapters import get_adapter
    model = get_adapter('timesfm25').model_id
    cases = {'a': _case('a', (32, 32), -2.0, horizon=129),
             'b': _case('b', (31, 33, 32), 3.0, horizon=128, quantiles=(0.2, 0.8))}
    offline, offline_states = _offline(tmp_path, cases, monkeypatch)
    evidence = {'offline_cycles': 3, 'offline_failures': 9,
                'offline_states': offline_states, 'states': []}
    with _server(tmp_path / 'http', gpu) as (url, control, log, gate, child):
        for label, data in cases.items():
            _compare(_post(url, model, f'parity-{label}', data), offline[label], data)
        evidence['states'].append(_quiescent(control, url))
        for cycle in range(3):
            # External IDs may repeat concurrently; internal wire IDs must not.
            with ThreadPoolExecutor(max_workers=2) as clients:
                pending = [(label, clients.submit(_post, url, model, 'reused-client-id', data))
                           for label, data in cases.items()]
                for label, future in pending:
                    _compare(future.result(timeout=60), offline[label], cases[label])
            for trigger in ('__fail_pre__', '__fail_submit__', '__fail_post__'):
                response = _request(url, model, f'fail-{cycle}-{trigger}',
                                    _case(trigger, (32,), -1.0))
                assert 400 <= response.status_code < 600, response.text
                _compare(_post(url, model, f'healthy-{cycle}-{trigger}', cases['a']),
                         offline['a'], cases['a'])
                evidence['states'].append(_quiescent(control, url))

            control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
            connection = None
            try:
                cancel_id = f'queued-disconnect-{cycle}'
                connection = _disconnectable(url, model, cancel_id, cases['a'])
                _wait_log(log, f'Added request pooling-{cancel_id}-')
                connection.shutdown(socket.SHUT_RDWR)
                connection.close()
                connection = None
                _wait_log(log, f'Request pooling-{cancel_id}')
                _wait(lambda: _api_state(control, url)['api_unfinished'] == 0,
                      'Disconnected queued request remained registered')
            finally:
                if connection is not None:
                    connection.close()
                control.post(url + '/resume', timeout=10).raise_for_status()
            _compare(_post(url, model, f'after-queued-{cycle}', cases['a']), offline['a'], cases['a'])
            evidence['states'].append(_quiescent(control, url))

            # A client read timeout is another disconnect, before execution.
            import requests
            control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
            try:
                with ThreadPoolExecutor(max_workers=1) as clients:
                    timed_out = clients.submit(_request, url, model, f'timeout-{cycle}',
                                               cases['a'], timeout=2.0)
                    _wait_for_submission(log, f'timeout-{cycle}', timed_out)
                    with pytest.raises(requests.exceptions.ReadTimeout):
                        timed_out.result(timeout=10)
                _wait(lambda: _api_state(control, url)['api_unfinished'] == 0,
                      'Timed-out request remained registered')
            finally:
                control.post(url + '/resume', timeout=10).raise_for_status()
            _compare(_post(url, model, f'after-timeout-{cycle}', cases['a']), offline['a'], cases['a'])
            evidence['states'].append(_quiescent(control, url))

            for marker in ('entered', 'release'):
                (gate / marker).unlink(missing_ok=True)
            # The worker has entered model.forward but has not launched TimesFM
            # kernels: this checks cancellation while worker-active, not CUDA preemption.
            gated = _case('__gate__', (32,), -2.0, horizon=129)
            connection = _disconnectable(url, model, f'inflight-{cycle}', gated)
            try:
                _wait(lambda: (gate / 'entered').exists(), 'Worker did not enter the forecast gate')
                with ThreadPoolExecutor(max_workers=1) as clients:
                    healthy = clients.submit(_post, url, model, f'neighbor-{cycle}', cases['b'])
                    _wait_for_submission(log, f'neighbor-{cycle}', healthy)
                    connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                    connection = None
                    _wait(lambda: _api_state(control, url)['api_unfinished'] == 1,
                          'In-flight cancellation did not release its API collector')
                    (gate / 'release').write_text('release')
                    _compare(healthy.result(timeout=60), offline['b'], cases['b'])
            finally:
                (gate / 'release').write_text('release during cleanup')
                if connection is not None:
                    connection.close()
            _compare(_post(url, model, f'after-inflight-{cycle}', cases['a']), offline['a'], cases['a'])
            evidence['states'].append(_quiescent(control, url))
            assert child.poll() is None
    evidence.update(timesfm_lifecycle='passed', cancellation_cycles=3,
                    checkpoint=model, execution='vLLM V1 FP32 eager', synthetic=True)
    (tmp_path / 'lifecycle-evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


@pytest.mark.parametrize('mode', ['dense-graph', 'paged-graph'])
def test_timesfm_graph_owner_survives_http_disconnect_and_max_shape_fallback(tmp_path, gpu, mode):
    from vtsfm.adapters import get_adapter

    model = get_adapter('timesfm25').model_id
    data = _case('graph', (512, 512), -1.5, horizon=1024)
    with _server(tmp_path / mode, gpu, timesfm_forecast_mode=mode) as (url, control, log, gate, child):
        initial = _rpc(control, url, 'bench_timesfm_forecast')
        assert initial['capture_attempts'] == 1 and initial['calls'] == 0
        baseline = _post(url, model, 'before-graph-disconnect', data)
        _quiescent(control, url)

        for phase in ('queued', 'inflight'):
            if phase == 'queued':
                control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
                cancel_data = data
            else:
                cancel_data = _case('__gate__', (512, 512), 3.0, horizon=1024)
            connection = None
            try:
                identity = f'graph-{phase}-disconnect'
                connection = _disconnectable(url, model, identity, cancel_data)
                if phase == 'queued':
                    _wait_log(log, f'Added request pooling-{identity}-')
                else:
                    _wait(lambda: (gate / 'entered').exists(), 'Graph request did not enter worker')
                connection.shutdown(socket.SHUT_RDWR)
                connection.close()
                connection = None
                _wait(lambda: _api_state(control, url)['api_unfinished'] == 0,
                      'Disconnected graph request remained registered')
            finally:
                if connection is not None:
                    connection.close()
                if phase == 'queued':
                    control.post(url + '/resume', timeout=10).raise_for_status()
                else:
                    (gate / 'release').write_text('release')
            _compare(_post(url, model, f'after-graph-{phase}', data), baseline, data)
            _quiescent(control, url)

        control.post(url + '/collective_rpc', json={'method': 'bench_memory_reset'}, timeout=20).raise_for_status()
        largest = _case('maximum-fallback', (8192,) * 16, .2, horizon=1024)
        output = _post(url, model, 'maximum-fallback', largest)
        assert len(output) == 16
        assert all(len(entry['0.5']) == 1024 for entry in output)
        memory = _rpc(control, url, 'bench_memory_stats')
        _compare(_post(url, model, 'after-maximum-fallback', data), baseline, data)
        _quiescent(control, url)
        final = _rpc(control, url, 'bench_timesfm_forecast')
        assert final['capture_attempts'] == 1 and final['fallbacks'] == 1
        assert final['graph_replays'] == final['calls'] == 5
        assert child.poll() is None
        (tmp_path / f'{mode}-evidence.json').write_text(json.dumps({
            'initial': initial, 'final': final, 'fallback_memory': memory,
            'queued_disconnect': 'passed', 'inflight_disconnect': 'passed',
        }, indent=2))


@pytest.mark.usefixtures('gpu')
def test_real_timesfm_model_exception_is_fatal_and_restart_recovers(tmp_path, gpu):
    from vtsfm.adapters import get_adapter
    import requests

    model = get_adapter('timesfm25').model_id
    case = _case('restart-healthy', (32,), -1.0)
    with _server(tmp_path / 'fault', gpu) as (url, control, log, gate, child):
        expected = _post(url, model, 'before-fatal', case)
        try:
            response = _request(url, model, 'fatal-model-error',
                                _case('__fail_model__', (32,), -1.0))
            assert response.status_code >= 500, response.text
        except requests.exceptions.ConnectionError:
            pass  # The failed engine can terminate its API process first.
        _wait(lambda: (gate / 'model-failure').exists(), 'Model fault was not injected')
        _wait_log(log, 'injected TimesFM model-execution failure')
        # A worker exception is engine-fatal in V1, not a recoverable request.
        _wait(lambda: child.poll() is not None, 'API process did not exit after engine death', timeout=30)
    with _server(tmp_path / 'restart', gpu) as (url, control, log, gate, child):
        _compare(_post(url, model, 'after-restart', case), expected, case)
        state = _quiescent(control, url)
    evidence = {'timesfm_model_failure': 'fail-stop, owned processes cleaned, restart passed',
                'state_after_restart': state, 'synthetic': True}
    (tmp_path / 'model-failure-evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))
