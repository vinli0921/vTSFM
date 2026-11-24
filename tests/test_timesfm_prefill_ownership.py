"""Manual HTTP check of TimesFM shared-prefill cache ownership and slot reuse."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket
import subprocess
import sys

import pytest

import test_timesfm_continuous as continuous
from model_integration_support import QUANTILES, compare_response
from test_timesfm_http_batching import _case, _post, _wait_for_submission
from test_timesfm_lifecycle import _disconnectable


gpu = continuous.gpu
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_PREFILL_OWNERSHIP') != '1',
    reason='Set VTSFM_RUN_TIMESFM_PREFILL_OWNERSHIP=1 on an idle TimesFM GPU')]


def _native_forecasts(cases, directory):
    # End the native CUDA process before vLLM claims the same GPU.
    path = directory / 'ownership-cases.json'
    output = directory / 'native-forecasts.json'
    path.write_text(json.dumps([{'name': name, 'data': data} for name, data in cases.items()]))
    subprocess.run([sys.executable, str(continuous.ROOT / 'tests/model_integration_support.py'),
                    '--adapter', 'timesfm25', '--cases', str(path), '--output', str(output)],
                   env={**os.environ, 'NVIDIA_TF32_OVERRIDE': '0', 'VTSFM_TEST_PRECISION': 'fp32'},
                   check=True, timeout=240)
    return {entry['case']['name']: entry for entry in json.loads(output.read_text())['cases']}


def _cycle(server, cycle, cancel, cases, reference):
    """Fill both slots together, retire one owner, refill before its peer ends."""
    peer = 'disconnect' if cancel else 'finish'
    survivor = 'survivor' if cancel else 'ragged'
    new = 'replacement'
    ids = {name: f'ownership-{cycle}-{name}' for name in (survivor, peer, new)}
    wire = {name: f'pooling-{value}' for name, value in ids.items()}
    parity = {}
    url, control, model = server['url'], server['control'], server['model']

    def check(name, future):
        predictions = future.result(timeout=60)
        parity[name] = compare_response({'data': {'predictions': predictions}},
                                        reference[name]['case'], reference[name]['expected'])

    continuous._arm(server, 'prefill')
    continuous._arm(server, 'decode')
    control.post(url + '/pause', params={'mode': 'keep'}, timeout=10).raise_for_status()
    connection = None
    with ThreadPoolExecutor(max_workers=3) as clients:
        try:
            live = clients.submit(_post, url, model, ids[survivor], cases[survivor])
            _wait_for_submission(server['log'], ids[survivor], live)
            if cancel:
                connection = _disconnectable(url, model, ids[peer], cases[peer])
                marker = f'Added request {wire[peer]}-'
                continuous._wait(lambda: marker in server['log'].read_text(errors='replace'),
                                 'Disconnectable peer did not enter EngineCore')
            else:
                finishing = clients.submit(_post, url, model, ids[peer], cases[peer])
                _wait_for_submission(server['log'], ids[peer], finishing)
            control.post(url + '/resume', timeout=10).raise_for_status()
            continuous._wait(lambda: (server['gate'] / 'prefill-entered').exists(),
                             'Initial shared prefill did not reach the gate')
            replacement = clients.submit(_post, url, model, ids[new], cases[new])
            _wait_for_submission(server['log'], ids[new], replacement)
            (server['gate'] / 'prefill-release').write_text('replacement queued')
            continuous._wait(lambda: (server['gate'] / 'decode-entered').exists(),
                             'The original owners did not run their first decode')
            if cancel:
                connection.shutdown(socket.SHUT_RDWR)
                connection.close()
                connection = None
                marker = f'Aborted request(s) {wire[peer]}-'
                continuous._wait(lambda: marker in server['log'].read_text(errors='replace'),
                                 'Client disconnect did not abort the active peer')
            (server['gate'] / 'decode-release').write_text('complete or cancel the peer')
            if not cancel:
                check(peer, finishing)
            check(new, replacement)
            check(survivor, live)
        finally:
            continuous._release(server)
            if connection is not None:
                connection.close()
            control.post(url + '/resume', timeout=10).raise_for_status()

    empty = continuous._empty(server)
    events = continuous._events(server['trace'])
    assert any(e.get('stage') == 'prefill' and set(e['request_ids']) == {wire[peer], wire[survivor]}
               for e in events)
    if cancel:
        assert any(e.get('reason') == 'cancel' and wire[peer] in e['request_ids'] for e in events)
    else:
        assert any(wire[peer] in e.get('finished_ids', []) for e in events)
    report = continuous._rpc(server, 'continuous_storage_report')
    admitted = next(e for e in report['admissions'] if e['new'] == [wire[new]])
    assert admitted['live_storages'][wire[peer]] == 0
    assert admitted['live_storages'][wire[survivor]] > 0
    assert all(count == 0 for count in report['live_storages'].values())
    owners = report['shared_owners']
    assert set(owners) == {wire[survivor], wire[peer]}
    for state in owners.values():
        assert state['kv_counts_by_flip'] == [1, 1]  # All K/V layers in each flip share one backing.
        assert state['counts_by_flip'] == [3, 3]  # Each flip also owns int32 position and int64 mask counters.
        assert state['backing_bytes'] == state['logical_bytes']
    assert report['disjoint'], 'Different owners or flip streams share a backing allocation'
    return {'retired_by': 'disconnect' if cancel else 'completion',
            'parity': parity, 'storage': report, 'empty_worker': empty}


def test_retired_shared_prefill_storage_expires_before_slot_reuse(gpu, tmp_path):
    cases = {name: _case(name, lengths, offset, horizon, QUANTILES)
             for name, lengths, offset, horizon in (
                 ('survivor', (512, 512), -1., 1024), ('disconnect', (512, 512), 2., 1024),
                 ('ragged', (511, 512), -2.5, 1023), ('finish', (497,), 3., 129),
                 ('replacement', (225, 257, 240), .5, 255))}
    reference = _native_forecasts(cases, tmp_path)
    gpu.wait_idle()
    cycles = []
    with continuous._server(tmp_path / 'ownership', gpu, slots=2, queue_limit=8, fusion_rows=16) as server:
        continuous._rpc(server, 'continuous_watch_prefill_storage')
        for index in range(6):
            cycles.append(_cycle(server, index, index % 2 == 0, cases, reference))
        assert server['child'].poll() is None
    (tmp_path / 'ownership-lifecycle.json').write_text(json.dumps(
        {'synthetic': True, 'cycles': cycles, 'gpu': gpu.metadata(), 'status': 'passed'}, indent=2) + '\n')
