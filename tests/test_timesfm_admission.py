"""Manual real-TimesFM request admission, execution limits and memory probes.

    CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_ADMISSION=1 PYTHONPATH=src:tests:. \\
      .venv-timesfm25/bin/python -m pytest tests/test_timesfm_admission.py -q -s

Synthetic inputs check correctness and the configured GPU envelope. The memory
samples are shape probes, not a throughput benchmark or exact memory guarantee.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest

import test_routing_engine
from test_timesfm_http_batching import _case, _compare, _post, _run_wave


ROOT = Path(__file__).resolve().parents[1]
gpu = test_routing_engine.gpu
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_ADMISSION') != '1',
    reason='Set VTSFM_RUN_TIMESFM_ADMISSION=1 on an idle Linux GPU')]


def _rpc(control, url, method, *args):
    response = control.post(url + '/collective_rpc', json={
        'method': method, 'args': list(args), 'timeout': 20}, timeout=25)
    response.raise_for_status()
    results = response.json()['results']
    assert len(results) == 1
    return results[0]


def _events(path):
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for event in events:
        event['request_ids'] = [identity.rsplit(':', 1)[0] for identity in event['request_ids']]
    return events


@contextmanager
def _server(directory, monitor, limits=None):
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
    url = f'http://127.0.0.1:{port}'
    config = directory / 'limits.json'
    if limits is not None:
        config.write_text(json.dumps(limits, indent=2))
    command, environment = build_vllm_command(
        adapter='timesfm25', port=port, max_num_seqs=4, forecast_batch_rows=16,
        forecast_limits=str(config) if limits is not None else None, benchmark=True)
    command[command.index('--worker-extension-cls') + 1] = 'admission_probe.AdmissionProbeExtension'
    command += ['--gpu-memory-utilization', '0.35', '--mm-processor-cache-gb', '0',
                '--enable-log-requests']
    execution_trace = directory / 'executions.jsonl'
    worker_trace = directory / 'worker-forwards.jsonl'
    environment.update(VTSFM_FORECAST_BATCH_TRACE=str(execution_trace),
                       NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4',
                       PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]))
    with OwnedProcesses(directory) as owner, requests.Session() as control:
        control.trust_env = False
        child = owner.spawn(command, environment, 'timesfm-admission-server')
        wait_ready(child, url, control, timeout=180)
        _rpc(control, url, 'admission_install', str(worker_trace))
        yield {'url': url, 'control': control, 'child': child,
               'log': directory / 'timesfm-admission-server.log',
               'execution_trace': execution_trace, 'worker_trace': worker_trace,
               'model': get_adapter('timesfm25').model_id}
    monitor.wait_idle()


def _reject(server, request_id, data, expected):
    response = server['control'].post(server['url'] + '/pooling',
        headers={'X-Request-Id': request_id},
        json={'model': server['model'], 'task': 'plugin', 'data': data}, timeout=30)
    assert response.status_code == 400, response.text
    assert expected in response.text.lower(), response.text
    assert f'Added request pooling-{request_id}-' not in server['log'].read_text(errors='replace')
    assert not any(f'pooling-{request_id}' in event['request_ids']
                   for event in _events(server['worker_trace']))
    return {'request_id': request_id, 'status': response.status_code,
            'error': response.json(), 'not_submitted': True}


def test_real_timesfm_rejects_before_submission_and_splits_aggregate_work(gpu, tmp_path):
    from vtsfm.adapters.timesfm25.admission import forecast_cost

    limits = {'max_request_rows': 4, 'max_context': 64, 'max_horizon': 256,
              'max_execution_rows': 8, 'max_execution_work': 200,
              'max_scheduled_requests': 4}
    cases = {'a': _case('a', (32, 32), -2.0),
             'b': _case('b', (32, 32), 3.0, quantiles=(0.2, 0.8)),
             'c': _case('c', (32, 32), 1.0, quantiles=(0.5,))}
    assert forecast_cost(2, 32, 129).work_units == 100
    rejected = {}
    with _server(tmp_path / 'tight', gpu, limits) as server:
        url, model = server['url'], server['model']
        isolated = {label: _post(url, model, f'isolated-{label}', data)
                    for label, data in cases.items()}
        for label, data, expected in (
            ('rows', _case('rows', (32,) * 5, 0, horizon=1), 'rows'),
            ('context', _case('context', (65,), 0, horizon=1), 'context'),
            ('horizon', _case('horizon', (32,), 0, horizon=257), 'horizon'),
            ('work', _case('work', (64,) * 3, 0, horizon=256), 'work'),
        ):
            rejected[label] = _reject(server, f'rejected-{label}', data, expected)
            _compare(_post(url, model, f'after-reject-{label}', cases['a']), isolated['a'], cases['a'])
            assert server['child'].poll() is None

        # Exactly-on-boundary cases remain supported: max rows, context, horizon
        # and 200 work units. Distinct shapes exercise independent limit checks.
        for label, data in (
            ('row-boundary', _case('row-boundary', (32,) * 4, 0, horizon=1)),
            ('context-boundary', _case('context-boundary', (64,), 0, horizon=1)),
            ('horizon-boundary', _case('horizon-boundary', (32,), 0, horizon=256)),
            ('work-boundary', _case('work-boundary', (32,) * 4, 0, horizon=129)),
        ):
            _post(url, model, label, data)

        completed = _run_wave(url, server['control'], model, server['log'],
                              'limited-wave', ('a', 'b', 'c'), cases)
        ids = {request_id for request_id, _, _ in completed}
        forwards = [event for event in _events(server['worker_trace']) if ids & set(event['request_ids'])]
        assert len(forwards) == 1 and set(forwards[0]['request_ids']) == ids
        executions = [event for event in _events(server['execution_trace']) if ids & set(event['request_ids'])]
        assert sorted(event['rows'] for event in executions) == [2, 4]
        assert sorted(identity for event in executions for identity in event['request_ids']) == sorted(ids)
        for event in executions:
            work = forecast_cost(event['rows'], event['context_length'], event['horizon']).work_units
            assert work <= limits['max_execution_work']
            assert event['rows'] <= limits['max_execution_rows']
            if 'execution_work' in event:
                assert event['execution_work'] == work
        for _, label, predictions in completed:
            _compare(predictions, isolated[label], cases[label])
        # Continuing arrivals traverse the same limit checks without pausing.
        with ThreadPoolExecutor(max_workers=4) as clients:
            pending = [(label, clients.submit(_post, url, model, f'rolling-{index}', cases[label]))
                       for index, label in enumerate(('a', 'b', 'c') * 3)]
            for label, future in pending:
                _compare(future.result(timeout=60), isolated[label], cases[label])
        assert server['child'].poll() is None
    evidence = {'timesfm_admission': 'passed', 'limits': limits, 'synthetic': True,
                'rejected': rejected, 'shared_worker_forward': forwards[0],
                'bounded_model_calls': executions}
    (tmp_path / 'admission-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'timesfm_admission': 'passed', 'rejections': len(rejected),
                      'shared_forward_requests': len(ids),
                      'execution_rows': [event['rows'] for event in executions]}))


def _measure(server, monitor, operation):
    from benchmarks.run import Sampler

    _rpc(server['control'], server['url'], 'bench_memory_empty_cache')
    baseline = _rpc(server['control'], server['url'], 'bench_memory_reset')
    with Sampler(monitor, interval=0.02) as sampler:
        output = operation()
        stats = _rpc(server['control'], server['url'], 'bench_memory_stats')
    assert stats['peak_allocated_bytes'] >= baseline['allocated_bytes'] > 0
    assert stats['peak_reserved_bytes'] >= stats['peak_allocated_bytes']
    return output, {'baseline': baseline, 'worker': stats,
                    'peak_nvml_used_bytes': max(sample[1] for sample in sampler.samples),
                    'nvml_samples': sampler.samples}


def test_real_timesfm_default_envelope_peak_memory(gpu, tmp_path):
    from dataclasses import asdict
    from vtsfm.adapters import get_adapter
    from vtsfm.adapters.timesfm25.admission import DEFAULT_LIMITS

    selected = get_adapter('timesfm25')
    record = {'kind': 'synthetic real-model HTTP memory envelope',
              'limits': asdict(DEFAULT_LIMITS), 'model': selected.model_id,
              'revision': selected.model_revision, 'gpu': gpu.metadata(),
              'packages': {name: importlib.metadata.version(name) for name in ('torch', 'vllm', 'timesfm')},
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (ROOT / 'src/vtsfm/admission.py',
                               ROOT / 'src/vtsfm/adapters/timesfm25/admission.py',
                               ROOT / 'src/vtsfm/adapters/pooling.py') if path.exists()},
              'precision': 'FP32 eager; TF32 disabled', 'nvml_interval_seconds': 0.02,
              'notes': ['NVML is sampled and may miss brief peaks.',
                        'Torch allocator and whole-device NVML memory are separate measures.',
                        'This is a finite envelope probe, not an exact memory guarantee or speedup benchmark.'],
              'profiles': []}
    small = _case('baseline', (128,), -1.0, horizon=96)
    maximum = _case('maximum', (DEFAULT_LIMITS.max_context,) * DEFAULT_LIMITS.max_request_rows,
                    -1.0, horizon=DEFAULT_LIMITS.max_horizon)
    with _server(tmp_path / 'default', gpu) as server:
        url, model = server['url'], server['model']
        _post(url, model, 'warmup', small)
        expected, profile = _measure(server, gpu,
            lambda: _post(url, model, 'baseline-measured', small))
        record['profiles'].append(dict(name='single-small', **profile))
        maximal, profile = _measure(server, gpu,
            lambda: _post(url, model, 'maximum-isolated', maximum))
        record['profiles'].append(dict(name='single-maximum', **profile))
        cases = {'maximum': maximum}
        completed, profile = _measure(server, gpu,
            lambda: _run_wave(url, server['control'], model, server['log'],
                              'maximum-wave', ('maximum',) * 4, cases))
        record['profiles'].append(dict(name='four-concurrent-maximum', **profile))
        ids = {request_id for request_id, _, _ in completed}
        forwards = [event for event in _events(server['worker_trace']) if ids & set(event['request_ids'])]
        assert len(forwards) == 1 and set(forwards[0]['request_ids']) == ids
        assert forwards[0]['rows'] == [DEFAULT_LIMITS.max_request_rows] * 4
        executions = [event for event in _events(server['execution_trace']) if ids & set(event['request_ids'])]
        assert len(executions) == 4
        assert all(event['rows'] == DEFAULT_LIMITS.max_request_rows for event in executions)
        for _, _, predictions in completed:
            _compare(predictions, maximal, maximum)
        _compare(_post(url, model, 'healthy-after-maximum', small), expected, small)
        assert server['child'].poll() is None
    record['status'] = 'passed'
    (tmp_path / 'memory-evidence.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'timesfm_memory_envelope': 'passed',
        'profiles': [{'name': profile['name'],
                      'peak_allocated_bytes': profile['worker']['peak_allocated_bytes'],
                      'peak_reserved_bytes': profile['worker']['peak_reserved_bytes'],
                      'peak_nvml_used_bytes': profile['peak_nvml_used_bytes']}
                     for profile in record['profiles']]}))
