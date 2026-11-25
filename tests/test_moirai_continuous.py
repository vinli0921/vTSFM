"""Manual A100 qualification for native Moirai quantile rounds over V1 HTTP."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

import test_routing_engine
from test_timesfm_http_batching import _case, _post, _wait_for_submission
from test_timesfm_lifecycle import _disconnectable
from test_timesfm_continuous import _arm, _empty, _events, _release, _rpc, _server, _wait

gpu = test_routing_engine.gpu
ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get('VTSFM_RUN_MOIRAI_CONTINUOUS') != '1',
    reason='Set VTSFM_RUN_MOIRAI_CONTINUOUS=1 on an idle Linux A100')]


def test_native_parity_staggered_arrivals_cancellation_and_capacity(gpu, tmp_path, monkeypatch):
    from model_integration_support import compare_response

    monkeypatch.setenv('VTSFM_EXPERIMENTAL_MOIRAI_BLOCKS', '1')
    levels = tuple(i / 10 for i in range(1,10))
    data = {
        'peer': _case('peer', (79, 81), -1., horizon=1024, quantiles=levels),
        'other': _case('other', (81,), 1., horizon=1024, quantiles=levels),
        'short': _case('short', (32,), 2., horizon=64, quantiles=(.2, .8)),
        'partial': _case('partial', (33,), 4., horizon=65, quantiles=levels),
        'max': _case('max', (7168,) * 8, -0.5, horizon=1024, quantiles=levels),
    }
    manifest = [{'name': label, 'data': value} for label, value in data.items()]
    cases, native = tmp_path/'cases.json', tmp_path/'native.json'
    cases.write_text(json.dumps(manifest))
    subprocess.run([sys.executable, str(ROOT/'tests/model_integration_support.py'), '--adapter', 'moirai2',
                    '--cases', str(cases), '--output', str(native)], check=True,
        env={**os.environ, 'VTSFM_TEST_PRECISION': 'fp32', 'NVIDIA_TF32_OVERRIDE': '0'})
    expected = {r['case']['name']: r['expected'] for r in json.loads(native.read_text())['cases']}
    checks = []

    def check(label, predictions):
        checks.append(compare_response({'data': {'predictions': predictions}}, {'name':label, 'data':data[label]},
                                       expected[label]))

    with _server(tmp_path/'worker', gpu, adapter='moirai2', slots=3, queue_limit=4) as server:
        def post(label, identity):
            return _post(server['url'], server['model'], identity, data[label])

        ctl, url = server['control'], server['url']
        # Start equal-round owners together, then enqueue short work behind the first
        # block. The trace records both fusion and independent short completion.
        _arm(server, 'prefill')
        with ThreadPoolExecutor(max_workers=3) as clients:
            ctl.post(url + '/pause', params={'mode':'keep'}, timeout=10).raise_for_status()
            pending = {}
            try:
                for label in ('peer', 'other'):
                    pending[label] = clients.submit(post, label, 'join-' + label)
                    _wait_for_submission(server['log'], 'join-' + label, pending[label])
                ctl.post(url+'/resume', timeout=10).raise_for_status()
                _wait(lambda: (server['gate']/'prefill-entered').exists(), 'Missing joint Moirai prefill')
                pending['short'] = clients.submit(post, 'short', 'join-short')
                _wait_for_submission(server['log'], 'join-short', pending['short'])
                _release(server)
                for label, future in pending.items():
                    check(label, future.result(timeout=120))
            finally:
                _release(server)
                ctl.post(url+'/resume', timeout=10).raise_for_status()
        _empty(server)
        steps = [e for e in _events(server['trace']) if e['event'] == 'step']
        parents = {'pooling-join-peer', 'pooling-join-other'}
        assert any(e['stage'] == 'decode' and set(e['request_ids']) == parents for e in steps)
        ends = {wire:e['end_ns'] for e in steps for wire in e['finished_ids']}
        assert ends['pooling-join-short'] < min(ends[k] for k in parents)

        _arm(server,'decode')
        connection = _disconnectable(url, server['model'], 'cancel-round', data['peer'])
        try:
            _wait(lambda: (server['gate']/'decode-entered').exists(), 'Moirai never entered a recursive round')
            with ThreadPoolExecutor(max_workers=1) as clients:
                waiting = clients.submit(post, 'other', 'surviving')
                try:
                    _wait_for_submission(server['log'], 'surviving', waiting)
                    connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                finally:
                    _release(server)
                check('other', waiting.result(timeout=120))
        finally:
            _release(server)
            connection.close()
        assert _empty(server)['tensor_bytes'] == 0
        assert any(e.get('reason') == 'cancel' and 'pooling-cancel-round' in e['request_ids']
                   for e in _events(server['trace']))
        bad = _case('invalid', (12,), 0., horizon=64, quantiles=levels)
        bad['inputs'][0]['target'][3] = None
        rejection = ctl.post(url+'/pooling', json={'model':server['model'], 'task':'plugin', 'data':bad}, timeout=20)
        assert 400 <= rejection.status_code < 500
        assert _empty(server)['tensor_bytes'] == 0
        check('partial', post('partial','slot-reused'))
        # Separate full-envelope ownership/peak check. All 8 rows participate in 16 rounds.
        _rpc(server, 'bench_memory_reset')
        peaks = []
        for i in range(3):
            check('max', post('max', f'max-{i}'))
            empty = _empty(server)
            assert not empty['tensor_bytes'] and not empty['cache_bytes']
            peaks.append(_rpc(server, 'bench_memory_stats'))
        assert len({s['allocated_bytes'] for s in peaks}) == 1
        (tmp_path/'moirai-block-evidence.json').write_text(json.dumps({'native_parity': checks,
            'maximum_wave_allocator': peaks, 'short_and_peer_completion_ns': ends}, indent=2))
