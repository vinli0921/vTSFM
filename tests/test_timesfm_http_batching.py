"""Opt-in real-checkpoint test of shared whole-forecast execution through HTTP."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import time

import pytest

from test_routing_engine import gpu  # noqa: F401 - reuse the idle-GPU fixture


ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get('VTSFM_RUN_TIMESFM_BATCHING') != '1',
                       reason='Set VTSFM_RUN_TIMESFM_BATCHING=1 on an idle Linux GPU'),
]


def _case(label, lengths, offset, horizon=129, quantiles=(0.1, 0.5, 0.9)):
    return {
        'inputs': [
            {'item_id': f'{label}-{row}',
             'target': [offset + row + ((step * 7 + row * 3) % 17 - 8) / 13
                        for step in range(length)]}
            for row, length in enumerate(lengths)
        ],
        'parameters': {'prediction_length': horizon, 'quantile_levels': list(quantiles)},
    }


def _post(url, model, request_id, data):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            url + '/pooling', headers={'X-Request-Id': request_id},
            json={'model': model, 'task': 'plugin', 'data': data}, timeout=60,
        )
        response.raise_for_status()
        result = response.json()
    assert result['request_id'] == f'pooling-{request_id}'
    return result['data']['predictions']


def _wait_for_submission(log, request_id, future):
    # AsyncLLM logs this only after awaiting add_request_async to EngineCore.
    # Observe each admission while paused instead of guessing with a burst delay.
    marker = f'Added request pooling-{request_id}-'
    deadline = time.monotonic() + 15
    while marker not in log.read_text(errors='replace'):
        if future.done():
            future.result()
            pytest.fail(f'{request_id} completed before the scheduler resumed')
        if time.monotonic() >= deadline:
            pytest.fail(f'EngineCore did not acknowledge submission of {request_id} in {log}')
        time.sleep(0.02)


def _run_wave(url, control, model, log, prefix, labels, cases):
    request_ids = [f'{prefix}-{index}' for index in range(len(labels))]
    with ThreadPoolExecutor(max_workers=len(labels)) as clients:
        control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
        try:
            pending = []
            for request_id, label in zip(request_ids, labels, strict=True):
                future = clients.submit(_post, url, model, request_id, cases[label])
                pending.append(future)
                _wait_for_submission(log, request_id, future)
            assert not any(future.done() for future in pending)
        finally:
            control.post(url + '/resume', timeout=10).raise_for_status()
        return [(f'pooling-{request_id}', label, future.result(timeout=60))
                for request_id, label, future in zip(request_ids, labels, pending, strict=True)]


def _compare(actual, expected, case):
    import numpy as np

    assert len(actual) == len(expected) == len(case['inputs'])
    levels = [str(level) for level in case['parameters']['quantile_levels']]
    horizon = case['parameters']['prediction_length']
    for row, baseline, item in zip(actual, expected, case['inputs'], strict=True):
        assert row['item_id'] == baseline['item_id'] == item['item_id']
        assert [key for key in row if key != 'item_id'] == levels
        for level in levels:
            assert len(row[level]) == horizon
            assert np.isfinite(row[level]).all()
            np.testing.assert_allclose(row[level], baseline[level], rtol=1e-4, atol=1e-4)


@pytest.mark.usefixtures('gpu')
def test_real_timesfm_http_requests_share_model_execution(tmp_path):
    import requests

    from benchmarks.run import OwnedProcesses, ensure_port_free, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ensure_port_free('127.0.0.1', port)
    url = f'http://127.0.0.1:{port}'
    model = get_adapter('timesfm25').model_id
    trace = tmp_path / 'executions.jsonl'
    log = tmp_path / 'timesfm-batching-server.log'
    command, environment = build_vllm_command(
        adapter='timesfm25', port=port, max_num_seqs=4, forecast_batch_rows=8, benchmark=True,
    )
    command += ['--gpu-memory-utilization', '0.35', '--enable-log-requests']
    environment.update(VTSFM_FORECAST_BATCH_TRACE=str(trace), NVIDIA_TF32_OVERRIDE='0',
                       PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT)]))
    cases = {
        'a': _case('a', (32, 32), -2.0),
        'b': _case('b', (32, 32, 32), 3.0, quantiles=(0.2, 0.8)),
        'width': _case('width', (33,), -0.5),
        'horizon': _case('horizon', (32,), 1.5, horizon=128),
        'ragged': _case('ragged', (31, 32), -1.0),
        'five-a': _case('five-a', (32,) * 5, -4.0),
        'five-b': _case('five-b', (32,) * 5, 4.0),
        'oversized': _case('oversized', (32,) * 9, 0.0),
    }
    waves = {
        'compatible': ('a', 'b'),
        'incompatible': ('a', 'width', 'horizon', 'ragged'),
        'cap': ('five-a', 'five-b', 'a', 'oversized'),
        # Eight queued requests span two four-slot scheduler forwards.
        'queued': ('a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'),
    }
    with OwnedProcesses(tmp_path) as owner, requests.Session() as control:
        control.trust_env = False
        child = owner.spawn(command, environment, 'timesfm-batching-server')
        wait_ready(child, url, control, timeout=180)
        isolated = {label: _post(url, model, f'isolated-{label}', data)
                    for label, data in cases.items()}
        completed = {
            name: _run_wave(url, control, model, log, name, labels, cases)
            for name, labels in waves.items()
        }
        # Keep four clients active through successive completions without
        # pausing the scheduler or controlling which requests share a forward.
        with ThreadPoolExecutor(max_workers=4) as clients:
            rolling = [
                (label, clients.submit(_post, url, model, f'rolling-{index}', cases[label]))
                for index, label in enumerate(('a', 'b', 'width', 'ragged') * 3)
            ]
            for label, future in rolling:
                _compare(future.result(timeout=60), isolated[label], cases[label])
        assert trace.exists(), f'No worker executions recorded; see {log}'
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        # Wire IDs are unique per invocation even for reused HTTP headers.
        # This test deliberately assigns distinct headers for trace attribution.
        for event in events:
            event['request_ids'] = [identity.rsplit(':', 1)[0] for identity in event['request_ids']]

    by_id = {request_id: label for wave in completed.values() for request_id, label, _ in wave}
    relevant = [event for event in events if any(rid in by_id for rid in event['request_ids'])]
    assert Counter(rid for event in relevant for rid in event['request_ids']) == Counter(by_id.keys())
    for event in relevant:
        members = [cases[by_id[rid]] for rid in event['request_ids']]
        assert event['rows'] == sum(len(member['inputs']) for member in members)
        assert all(event['horizon'] == member['parameters']['prediction_length'] for member in members)
        assert all(event['context_length'] == max(len(item['target']) for item in member['inputs'])
                   for member in members)
        if event['kind'] == 'batch':
            assert 2 <= len(members) <= 4 and event['rows'] <= 8
        else:
            assert event['kind'] == 'single' and len(members) == 1

    def wave_events(name):
        return [event for event in relevant if event['request_ids'][0].startswith(f'pooling-{name}-')]

    assert [(event['kind'], event['rows']) for event in wave_events('compatible')] == [('batch', 5)]
    assert [event['kind'] for event in wave_events('incompatible')] == ['single'] * 4
    assert sorted(event['rows'] for event in wave_events('cap')) == [5, 7, 9]
    assert [event['rows'] for event in wave_events('cap') if event['kind'] == 'batch'] == [7]
    assert [(event['kind'], event['rows']) for event in wave_events('queued')] == [('batch', 8)] * 2
    for wave in completed.values():
        for _, label, predictions in wave:
            _compare(predictions, isolated[label], cases[label])
    print(json.dumps({'timesfm_http_batching': 'passed',
                      'http_requests': len(isolated) + len(by_id) + len(rolling),
                      'batched_executions': sum(event['kind'] == 'batch' for event in relevant),
                      'trace': str(trace)}))
