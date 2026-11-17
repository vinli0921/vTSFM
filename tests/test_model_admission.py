"""Manual real-checkpoint admission/fusion/memory checks for Toto and Moirai.

Run once in each adapter's Linux environment, on an idle GPU:

    CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ADMISSION_ADAPTER=toto2 PYTHONPATH=src:tests:. \\
      .venv-toto2/bin/python -m pytest tests/test_model_admission.py -q -s

Replace toto2 with moirai2 for the other adapter. Inputs are deterministic,
synthetic correctness and shape probes, not forecast-quality or speed evidence.
No weights or GPU work are loaded by normal CPU collection.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from importlib import import_module
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import pytest

from model_integration_support import compare_response
from model_serving_support import child_environment, save_json, source_metadata
import test_routing_engine
from test_timesfm_admission import _events, _measure, _reject, _rpc
from test_timesfm_http_batching import _case, _compare, _wait_for_submission


ROOT = Path(__file__).resolve().parents[1]
SELECTED = os.environ.get('VTSFM_RUN_ADMISSION_ADAPTER')
gpu = test_routing_engine.gpu
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    SELECTED is None,
    reason='Set VTSFM_RUN_ADMISSION_ADAPTER=toto2 or moirai2 on an idle Linux GPU')]


@pytest.fixture
def adapter():
    if SELECTED not in ('toto2', 'moirai2'):
        pytest.fail(f'Unknown admission adapter {SELECTED!r}; select toto2 or moirai2')
    return SELECTED


def _policy(adapter):
    module = import_module(f'vtsfm.adapters.{adapter}.admission')
    return module.DEFAULT_LIMITS, module.forecast_cost


def test_direct_model_launch_rejects_before_runtime_construction(adapter, monkeypatch):
    """Checkpoint-free: a direct vLLM caller must not bypass startup admission."""
    from vtsfm.adapters import get_adapter
    from vtsfm.admission import FORECAST_LIMITS_ENV
    from vtsfm.stepping import CONTINUOUS_ENV

    defaults, _ = _policy(adapter)
    module, name = get_adapter(adapter).model_class.split(':')
    model_class = getattr(import_module(module), name)

    def forbid_model_load(*args, **kwargs):
        pytest.fail('Unsafe launch reached runtime/model construction')

    monkeypatch.setattr(model_class, 'runtime_class', forbid_model_load)
    monkeypatch.delenv(CONTINUOUS_ENV, raising=False)
    monkeypatch.delenv(FORECAST_LIMITS_ENV, raising=False)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(
        max_num_seqs=defaults.max_scheduled_requests + 1))
    with pytest.raises(ValueError, match='scheduled requests'):
        model_class(vllm_config=config)
    config.scheduler_config.max_num_seqs = defaults.max_scheduled_requests
    monkeypatch.setenv(FORECAST_LIMITS_ENV, '{"max_forward_bytes":1}')
    with pytest.raises(ValueError, match='resident bytes'):
        model_class(vllm_config=config)


def test_offline_rejection_precedes_context_staging(adapter, monkeypatch):
    """Checkpoint-free: parse and direct preprocessing reject before torch.full."""
    import torch
    from vtsfm.adapters import get_adapter
    from vtsfm.admission import FORECAST_LIMITS_ENV
    from vtsfm.stepping import CONTINUOUS_ENV

    selected = get_adapter(adapter)
    module, name = selected.io_processor_class.rsplit('.', 1)
    monkeypatch.delenv(CONTINUOUS_ENV, raising=False)
    monkeypatch.setenv(FORECAST_LIMITS_ENV, '{"max_request_rows":1}')
    processor = getattr(import_module(module), name)(None, None)
    data = _case('over-rows', (selected.execution_capabilities.input_patch_size,) * 2,
                 0., horizon=1, quantiles=(.5,))

    def forbid_context_allocation(*args, **kwargs):
        pytest.fail('Rejected request allocated a context tensor')

    monkeypatch.setattr(torch, 'full', forbid_context_allocation)
    with pytest.raises(ValueError, match='rows'):
        processor.parse_data(data)
    with pytest.raises(ValueError, match='rows'):
        processor.pre_process((data['inputs'], 1, [.5]))


def _post(url, model, request_id, data, *, timeout=180):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.post(url + '/pooling', headers={'X-Request-Id': request_id},
            json={'model': model, 'task': 'plugin', 'data': data}, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    assert body['request_id'] == f'pooling-{request_id}'
    return body['data']['predictions']


def _wave(server, prefix, labels, cases):
    """Queue known admissions while paused; do not assume simultaneous arrivals."""
    control, url, model = server['control'], server['url'], server['model']
    request_ids = [f'{prefix}-{index}' for index in range(len(labels))]
    with ThreadPoolExecutor(max_workers=len(labels)) as clients:
        control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
        try:
            pending = []
            for request_id, label in zip(request_ids, labels, strict=True):
                future = clients.submit(_post, url, model, request_id, cases[label])
                pending.append(future)
                _wait_for_submission(server['log'], request_id, future)
            assert not any(future.done() for future in pending)
        finally:
            control.post(url + '/resume', timeout=10).raise_for_status()
        return [(f'pooling-{request_id}', label, future.result(timeout=240))
                for request_id, label, future in zip(request_ids, labels, pending, strict=True)]


def _empty(server):
    fields = ('worker_requests', 'worker_slots', 'worker_encoder_cache',
              'worker_pooling_params', 'worker_pooling_states',
              'worker_execute_state', 'pooler_routes')
    deadline = time.monotonic() + 15
    while True:
        state = _rpc(server['control'], server['url'], 'admission_snapshot')
        if not any(state[key] for key in fields):
            return state
        if time.monotonic() >= deadline:
            pytest.fail(f'Completed request state was retained: {state}')
        time.sleep(.02)


@contextmanager
def _server(directory, monitor, adapter, *, limits=None):
    import requests
    from benchmarks.run import OwnedProcesses, ensure_port_free, ensure_worker_gpu, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command

    defaults, _ = _policy(adapter)
    directory.mkdir(parents=True, exist_ok=True)
    monitor.wait_idle()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ensure_port_free('127.0.0.1', port)
    url = f'http://127.0.0.1:{port}'
    config = directory / 'limits.json'
    if limits is not None:
        save_json(config, limits)
    command, environment = build_vllm_command(
        adapter=adapter, port=port, max_num_seqs=4,
        forecast_batch_rows=defaults.max_execution_rows,
        forecast_limits=str(config) if limits is not None else None, benchmark=True)
    command[command.index('--worker-extension-cls') + 1] = 'admission_probe.AdmissionProbeExtension'
    command += ['--gpu-memory-utilization', '0.35', '--mm-processor-cache-gb', '0',
                '--enable-log-requests', '--no-async-scheduling']
    execution_trace = directory / 'executions.jsonl'
    worker_trace = directory / 'worker-forwards.jsonl'
    child_env = child_environment(monitor.index)
    for key in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE',
                'OMP_NUM_THREADS', 'PYTHONUNBUFFERED', 'PYTHONPATH'):
        environment[key] = child_env[key]
    environment['VTSFM_FORECAST_BATCH_TRACE'] = str(execution_trace)
    selected = get_adapter(adapter)
    save_json(directory / 'launch.json', {
        'command': command, 'adapter': adapter,
        'limits': json.loads(environment['VTSFM_FORECAST_LIMITS']),
        'model_id': selected.model_id, 'model_revision': selected.model_revision,
        'precision': 'eager FP32; TF32 disabled; pinned upstream attention',
    })
    label = f'{adapter}-admission-server'
    with OwnedProcesses(directory) as owner, requests.Session() as control:
        control.trust_env = False
        child = owner.spawn(command, environment, label)
        wait_ready(child, url, control, timeout=600)
        _rpc(control, url, 'admission_install', str(worker_trace))
        worker_config = _rpc(control, url, 'bench_memory_config')
        ensure_worker_gpu(worker_config, monitor.uuid)
        save_json(directory / 'worker-config.json', worker_config)
        yield {'url': url, 'control': control, 'child': child,
               'log': directory / f'{label}.log', 'model': selected.model_id,
               'execution_trace': execution_trace, 'worker_trace': worker_trace}
    monitor.wait_idle()


def _native(directory, monitor, adapter, cases):
    """Use a separate official-model process, then release it before serving."""
    from benchmarks.run import OwnedProcesses, ensure_worker_gpu
    from vtsfm.adapters import get_adapter

    directory.mkdir(parents=True, exist_ok=True)
    manifest, output = directory / 'cases.json', directory / 'native.json'
    entries = [{'name': label, 'data': data} for label, data in cases.items()]
    save_json(manifest, {'cases': entries, 'provenance': {'input_kind': 'synthetic'}})
    monitor.assert_idle()
    environment = child_environment(monitor.index)
    environment['VLLM_PLUGINS'] = ''
    with OwnedProcesses(directory) as owner:
        child = owner.spawn([sys.executable, str(ROOT / 'tests/model_integration_support.py'),
            '--adapter', adapter, '--cases', str(manifest), '--output', str(output)],
            environment, 'native-admission-reference')
        child.wait(timeout=900)
        assert child.returncode == 0, directory / 'native-admission-reference.log'
    monitor.wait_idle()
    result = json.loads(output.read_text())
    selected = get_adapter(adapter)
    assert (result['adapter'], result['model_id'], result['model_revision']) == (
        adapter, selected.model_id, selected.model_revision)
    ensure_worker_gpu({'gpu': {'uuid': result['runtime']['uuid']}}, monitor.uuid)
    assert [entry['case'] for entry in result['cases']] == entries
    return {entry['case']['name']: entry['expected'] for entry in result['cases']}


def _bounded_calls(server, completed, limits, forecast_cost):
    ids = {request_id for request_id, _, _ in completed}
    forwards = [event for event in _events(server['worker_trace']) if ids & set(event['request_ids'])]
    executions = [event for event in _events(server['execution_trace']) if ids & set(event['request_ids'])]
    assert Counter(identity for event in forwards for identity in event['request_ids']) == Counter(ids)
    assert Counter(identity for event in executions for identity in event['request_ids']) == Counter(ids)
    for event in executions:
        cost = forecast_cost(event['rows'], event['context_length'], event['horizon'])
        assert event['rows'] <= limits['max_execution_rows']
        assert cost.work_units <= limits['max_execution_work']
        assert event['execution_work'] == cost.work_units
    return forwards, executions


def test_real_model_frontend_boundaries_work_split_and_fused_routing(adapter, gpu, tmp_path):
    """Catch admission after submission, ignored aggregate budgets or misrouted fusion."""
    from vtsfm.adapters import get_adapter

    defaults, forecast_cost = _policy(adapter)
    capabilities = get_adapter(adapter).execution_capabilities
    context = capabilities.input_patch_size * 2
    horizon = capabilities.output_block_size + 1
    assert defaults.max_request_rows >= 2 and defaults.max_execution_rows >= 2
    assert context * 8 < defaults.max_context and horizon <= defaults.max_horizon
    # Two one-row requests fit exactly; the third must run separately despite
    # ample row capacity. This tests the adapter's own work proxy, not its formula.
    limits = {
        'max_request_rows': min(4, defaults.max_request_rows),
        'max_context': context * 8,
        'max_horizon': horizon,
        'max_execution_rows': min(8, defaults.max_execution_rows),
        'max_execution_work': forecast_cost(2, context, horizon).work_units,
        'max_scheduled_requests': 4,
    }
    assert limits['max_execution_work'] <= defaults.max_execution_work
    cases = {
        'a': _case('a', (context,), -2., horizon=horizon),
        'b': _case('b', (context,), 3., horizon=horizon, quantiles=(.2, .8)),
        'c': _case('c', (context,), 1., horizon=horizon, quantiles=(.5,)),
        'width': _case('width', (context + 1,), -.5, horizon=horizon),
        'horizon': _case('horizon', (context,), 1.5, horizon=1),
        'ragged': _case('ragged', (context - 1, context), -1., horizon=1),
    }
    references = _native(tmp_path / 'native', gpu, adapter, cases)
    evidence = {'adapter': adapter, 'synthetic': True, 'limits': limits,
                'native_parity': [], 'rejected': {}, 'status': 'failed'}
    try:
        with _server(tmp_path / 'tight', gpu, adapter, limits=limits) as server:
            url, model = server['url'], server['model']
            isolated = {label: _post(url, model, f'isolated-{label}', data)
                        for label, data in cases.items()}
            for label, data in cases.items():
                evidence['native_parity'].append(compare_response(
                    {'data': {'predictions': isolated[label]}}, {'name': label, 'data': data}, references[label]))

            work_rows = limits['max_request_rows']
            assert forecast_cost(work_rows, limits['max_context'], limits['max_horizon']).work_units > limits['max_execution_work']
            malformed = _case('malformed', (context,), 0., horizon=1)
            malformed['inputs'][0]['target'][0] = True
            for label, data, expected in (
                ('rows', _case('rows', (context,) * (limits['max_request_rows'] + 1), 0., horizon=1), 'rows'),
                ('context', _case('context', (limits['max_context'] + 1,), 0., horizon=1), 'context'),
                ('horizon', _case('horizon', (context,), 0., horizon=limits['max_horizon'] + 1), 'horizon'),
                ('work', _case('work', (limits['max_context'],) * work_rows, 0., horizon=limits['max_horizon']), 'work'),
                ('malformed', malformed, 'finite'),
            ):
                evidence['rejected'][label] = _reject(server, f'rejected-{label}', data, expected)
                _compare(_post(url, model, f'after-reject-{label}', cases['a']), isolated['a'], cases['a'])
                assert server['child'].poll() is None

            boundaries = {
                'rows': _case('row-boundary', (context,) * limits['max_request_rows'], 0., horizon=1),
                'context': _case('context-boundary', (limits['max_context'],), 0., horizon=1),
                'horizon': _case('horizon-boundary', (context,), 0., horizon=limits['max_horizon']),
                'work': _case('work-boundary', (context,) * 2, 0., horizon=horizon),
            }
            for label, data in boundaries.items():
                _post(url, model, f'{label}-boundary', data)
            completed = _wave(server, 'limited-wave', ('a', 'b', 'c'), cases)
            forwards, executions = _bounded_calls(server, completed, limits, forecast_cost)
            assert len(forwards) == 1 and len(forwards[0]['request_ids']) == 3
            assert sorted(event['rows'] for event in executions) == [1, 2]
            assert [(event['kind'], event['rows']) for event in executions if len(event['request_ids']) > 1] == [('batch', 2)]
            evidence.update(shared_worker_forward=forwards[0], bounded_model_calls=executions)
            mixed = _wave(server, 'mixed-wave', ('a', 'width', 'horizon', 'ragged'), cases)
            _bounded_calls(server, mixed, limits, forecast_cost)
            for _, label, predictions in completed + mixed:
                _compare(predictions, isolated[label], cases[label])
                compare_response({'data': {'predictions': predictions}},
                    {'name': label, 'data': cases[label]}, references[label])
            with ThreadPoolExecutor(max_workers=4) as clients:
                rolling = [(label, clients.submit(_post, url, model, f'rolling-{index}', cases[label]))
                           for index, label in enumerate(('a', 'b', 'width', 'ragged') * 3)]
                for label, future in rolling:
                    _compare(future.result(timeout=180), isolated[label], cases[label])
            evidence['empty_request_state'] = _empty(server)
            assert server['child'].poll() is None
        evidence['status'] = 'passed'
    finally:
        save_json(tmp_path / 'admission-evidence.json', evidence)
    print(json.dumps({'admission_adapter': adapter, 'status': 'passed',
                      'rejections': len(evidence['rejected']), 'execution_rows': [1, 2]}))


def test_real_model_default_envelope_repeated_maximum_memory(adapter, gpu, tmp_path):
    """Catch unsafe default corners, fusion over-budget or retained request state."""
    from vtsfm.adapters import get_adapter

    defaults, forecast_cost = _policy(adapter)
    selected = get_adapter(adapter)
    # A tighter work ceiling may exclude the rectangular shape-limit corner.
    # Select the largest admitted row count at the maximum context and horizon.
    rows = min(defaults.max_request_rows, defaults.max_execution_rows)
    while rows:
        try:
            defaults.validate_request(forecast_cost(rows, defaults.max_context, defaults.max_horizon))
            break
        except ValueError:
            rows -= 1
    assert rows, 'The published policy admits no row at its maximum context/horizon'
    maximum_cost = forecast_cost(rows, defaults.max_context, defaults.max_horizon)
    assert defaults.max_execution_rows % 4 == 0
    fused_rows = defaults.max_execution_rows // 4
    assert 1 <= fused_rows <= defaults.max_request_rows
    defaults.validate_request(forecast_cost(fused_rows, defaults.max_context, defaults.max_horizon))
    assert forecast_cost(defaults.max_execution_rows, defaults.max_context,
                         defaults.max_horizon).work_units <= defaults.max_execution_work
    sources = [ROOT / 'src/vtsfm/admission.py', ROOT / 'src/vtsfm/adapters/pooling.py',
               ROOT / 'src/vtsfm/adapters/execution_batch.py',
               ROOT / f'src/vtsfm/adapters/{adapter}/admission.py',
               ROOT / f'src/vtsfm/adapters/{adapter}/runtime.py', Path(__file__)]
    package = {'toto2': 'toto-2', 'moirai2': 'uni2ts'}[adapter]
    packages = ['torch', 'vllm', package]
    if adapter == 'moirai2':
        packages.append('lightning')
    record = {
        **source_metadata(), 'kind': 'synthetic real-model HTTP memory envelope',
        'adapter': adapter, 'limits': asdict(defaults), 'maximum_cost': asdict(maximum_cost),
        'fused_rows_per_request': fused_rows,
        'model_id': selected.model_id, 'model_revision': selected.model_revision,
        'gpu': gpu.metadata(),
        'packages': {name: importlib.metadata.version(name) for name in packages},
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        'precision': 'eager FP32; TF32 disabled; pinned upstream attention',
        'nvml_interval_seconds': .02, 'status': 'failed', 'profiles': [],
        'notes': ['NVML samples may miss brief peaks; allocator and device memory differ.',
                  'The corner maximizes admitted rows at maximum context and horizon.',
                  'Upstream model-owned high-water KV storage may remain after forecasts.',
                  'Request-owned state must clear; repeated same-shape waves must not grow allocations.',
                  'Finite shape probes do not establish an exact memory guarantee or speedup.'],
    }
    small = _case('small', (128,), -1., horizon=96)
    maximum = _case('maximum', (defaults.max_context,) * rows, -1., horizon=defaults.max_horizon)
    fused = _case('fused-maximum', (defaults.max_context,) * fused_rows, -1.,
                  horizon=defaults.max_horizon)
    try:
        native = _native(tmp_path / 'native-fused-maximum', gpu, adapter, {'fused-maximum': fused})
        record['fused_native_reference'] = 'native-fused-maximum/native.json'
        with _server(tmp_path / 'default', gpu, adapter) as server:
            url, model = server['url'], server['model']
            _post(url, model, 'warmup', small)
            expected, profile = _measure(server, gpu, lambda: _post(url, model, 'small-measured', small))
            record['profiles'].append(dict(name='single-small', **profile))
            maximal, profile = _measure(server, gpu, lambda: _post(url, model, 'maximum-isolated', maximum))
            record['profiles'].append(dict(name='single-maximum', **profile))
            record['after_single_maximum'] = _empty(server)
            fused_isolated = _post(url, model, 'fused-maximum-isolated', fused)
            fused_case = {'name': 'fused-maximum', 'data': fused}
            record['fused_native_parity'] = [compare_response(
                {'data': {'predictions': fused_isolated}}, fused_case, native['fused-maximum'])]
            completed, profile = _measure(server, gpu,
                lambda: _wave(server, 'fused-maximum-wave', ('fused',) * 4, {'fused': fused}))
            forwards, executions = _bounded_calls(server, completed, asdict(defaults), forecast_cost)
            assert len(forwards) == 1 and forwards[0]['rows'] == [fused_rows] * 4
            assert len(executions) == 1, 'The maximum fused execute path was not exercised'
            assert executions[0]['kind'] == 'batch'
            assert executions[0]['rows'] == defaults.max_execution_rows
            assert len(executions[0]['request_ids']) == 4
            for _, _, predictions in completed:
                _compare(predictions, fused_isolated, fused)
                record['fused_native_parity'].append(compare_response(
                    {'data': {'predictions': predictions}}, fused_case, native['fused-maximum']))
            profile['worker_forwards'], profile['model_calls'] = forwards, executions
            record['profiles'].append(dict(name='four-request-maximum-fused-call', **profile))
            record['after_maximum_fused_call'] = _empty(server)
            snapshots = []
            for index in range(3):
                completed, profile = _measure(server, gpu,
                    lambda: _wave(server, f'maximum-wave-{index}', ('maximum',) * 4, {'maximum': maximum}))
                forwards, executions = _bounded_calls(server, completed, asdict(defaults), forecast_cost)
                assert len(forwards) == 1 and forwards[0]['rows'] == [rows] * 4
                for _, _, predictions in completed:
                    _compare(predictions, maximal, maximum)
                profile['worker_forwards'], profile['model_calls'] = forwards, executions
                record['profiles'].append(dict(name=f'four-concurrent-maximum-{index}', **profile))
                state = _empty(server)
                snapshots.append(state)
                record['empty_request_states'] = snapshots
                assert server['child'].poll() is None
            # Compare only after the first maximum wave has established any
            # model-owned high-water cache, not against the small-model baseline.
            assert max(state['allocated_bytes'] for state in snapshots[1:]) <= snapshots[0]['allocated_bytes'] + 64 * 1024
            _compare(_post(url, model, 'healthy-after-maximum', small), expected, small)
            record['final_request_state'] = _empty(server)
        record['status'] = 'passed'
    finally:
        save_json(tmp_path / 'memory-evidence.json', record)
    print(json.dumps({'admission_memory_adapter': adapter, 'status': 'passed',
        'maximum_shape': [rows, defaults.max_context, defaults.max_horizon],
        'profiles': [{'name': profile['name'],
                      'peak_allocated_bytes': profile['worker']['peak_allocated_bytes'],
                      'peak_reserved_bytes': profile['worker']['peak_reserved_bytes'],
                      'peak_nvml_used_bytes': profile['peak_nvml_used_bytes']}
                     for profile in record['profiles']]}))
