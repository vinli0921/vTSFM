"""Manual real HTTP cancellation of a waiter held by prefill grouping.

Run with VTSFM_RUN_TIMESFM_PREFILL_HOLD=1 in the pinned TimesFM environment.
Worker gates place disconnects at the scheduling boundary deterministically.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket

import pytest

import test_timesfm_continuous as continuous
from test_timesfm_http_batching import _case, _compare, _post, _wait_for_submission
from test_timesfm_lifecycle import _disconnectable


gpu = continuous.gpu
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_TIMESFM_PREFILL_HOLD') != '1',
    reason='Set VTSFM_RUN_TIMESFM_PREFILL_HOLD=1 in the candidate TimesFM environment')]


def _case9(label, rows, horizon, offset):
    return _case(label, (512,) * rows, offset, horizon=horizon,
                 quantiles=tuple(level / 10 for level in range(1, 10)))


def _wait_for_admission(server, request_id):
    marker = f'Added request pooling-{request_id}-'
    continuous._wait(lambda: marker in server['log'].read_text(errors='replace'),
                     f'Request was not submitted: {request_id}')


def _held_events(server, identity):
    return [event for event in continuous._events(server['trace'])
            if event.get('event') == 'prefill_group_hold' and identity in event.get('request_ids', [])]


@pytest.mark.parametrize('budget_steps', [1, 2])
def test_cancel_during_hold_releases_waiter_and_preserves_healthy_forecasts(gpu, tmp_path, monkeypatch, budget_steps):
    from benchmarks.continuous import _drain_requests

    active = _case9('finishing-active', 1, 256, -1.)
    queued = {label: _case9(label, 2, 1024, offset)
              for label, offset in (('c', 1.), ('d', 2.), ('e', 3.))}
    cancelled = _case9('cancelled', 2, 1024, 4.)
    lone = _case9('lone', 1, 1024, -.5)
    short = _case9('short', 1, 96, .5)
    with continuous._server(tmp_path / 'whole-reference', gpu, continuous=False, slots=1) as server:
        expected_active = _post(server['url'], server['model'], 'reference-active', active)
        expected = {label: _post(server['url'], server['model'], f'reference-{label}', data)
                    for label, data in queued.items()}
        expected_lone = _post(server['url'], server['model'], 'reference-lone', lone)
        expected_short = _post(server['url'], server['model'], 'reference-short', short)
    evidence = {'synthetic': True, 'policy_steps': budget_steps,
                'reference': 'isolated eager whole-forecast HTTP with identical checkpoint/FP32 settings',
                'quantiles': [level / 10 for level in range(1, 10)], 'cycles': []}
    monkeypatch.setenv('VTSFM_PREFILL_GROUP_STEPS', str(budget_steps))
    with continuous._server(tmp_path / 'held-cancellation', gpu, slots=4, queue_limit=128, fusion_rows=16) as server:
        for cycle in range(3):
            continuous._arm(server, 'prefill')
            continuous._arm(server, 'decode')
            connection = None
            cancelled_id = f'held-cancel-{cycle}'
            active_id = f'held-active-{cycle}'
            with ThreadPoolExecutor(max_workers=4) as clients:
                try:
                    active_result = clients.submit(_post, server['url'], server['model'], active_id, active)
                    continuous._wait(lambda: (server['gate'] / 'prefill-entered').exists(),
                                     'Active request did not reach the prefill gate')
                    # The active owner has one block left. Four long compatible
                    # waiters could form a four-owner prefill after it finishes,
                    # instead of the three-owner cohort that fits immediately.
                    connection = _disconnectable(server['url'], server['model'], cancelled_id, cancelled)
                    _wait_for_admission(server, cancelled_id)
                    neighbors = {}
                    for label, data in queued.items():
                        request_id = f'held-neighbor-{cycle}-{label}'
                        neighbors[label] = clients.submit(_post, server['url'], server['model'], request_id, data)
                        _wait_for_submission(server['log'], request_id, neighbors[label])
                    (server['gate'] / 'prefill-release').write_text('Four long waiters submitted')
                    identity = 'pooling-' + cancelled_id
                    holds = continuous._wait(lambda: _held_events(server, identity),
                                             'No policy hold observed for the oldest waiter')
                    continuous._wait(lambda: (server['gate'] / 'decode-entered').exists(),
                                     'Existing decode did not progress during the hold')
                    events = continuous._events(server['trace'])
                    active_engine = next(event['engine_request_ids'][0] for event in events
                                         if event.get('event') == 'enqueue'
                                         and 'pooling-' + active_id in event.get('request_ids', []))
                    assert holds[0]['reason'] == 'near_completion'
                    assert holds[0]['budget_steps'] == budget_steps
                    assert active_engine in holds[0]['finishing_request_ids']
                    assert holds[0]['remaining_steps'][active_engine] == 1
                    connection.shutdown(socket.SHUT_RDWR)
                    connection.close()
                    connection = None
                    continuous._wait(lambda: f'Aborted request(s) pooling-{cancelled_id}-'
                        in server['log'].read_text(errors='replace'), 'Client disconnect did not abort held request')
                    (server['gate'] / 'decode-release').write_text('Held waiter disconnected; let active decode finish')
                    _compare(active_result.result(timeout=120), expected_active, active)
                    for label, future in neighbors.items():
                        _compare(future.result(timeout=120), expected[label], queued[label])
                finally:
                    continuous._release(server)
                    if connection is not None:
                        connection.close()
            empty = continuous._empty(server)
            drain = _drain_requests(server['url'], tmp_path / f'held-drain-{cycle}.json')
            events = continuous._events(server['trace'])
            assert not any(event.get('event') == 'step' and event.get('stage') == 'prefill'
                           and identity in event.get('request_ids', []) for event in events)
            cancelled_engine = next(event['engine_request_ids'][0] for event in events
                                    if event.get('event') == 'enqueue'
                                    and identity in event.get('request_ids', []))
            releases = [event for event in events if event.get('event') == 'prefill_group_release'
                        and event.get('waiter_engine_id') == cancelled_engine]
            assert len(releases) == 1 and releases[0]['reason'] == 'cancelled_or_replaced'
            _compare(_post(server['url'], server['model'], f'after-held-cancel-{cycle}', short), expected_short, short)
            evidence['cycles'].append({'cancelled_request_id': identity, 'holds': holds,
                'policy_release': releases[0], 'existing_decode_progressed': True,
                'cancelled_before_prefill': True, 'empty_worker': empty, 'empty_engine': drain})
        # Grouping must not invent a delay for an idle/lone request or a short
        # horizon. These are route/trace assertions, not timing thresholds.
        _compare(_post(server['url'], server['model'], 'lone-bypass', lone), expected_lone, lone)
        _compare(_post(server['url'], server['model'], 'short-bypass', short), expected_short, short)
        events = continuous._events(server['trace'])
        assert not any(event.get('event') == 'prefill_group_hold'
                       and {'pooling-lone-bypass', 'pooling-short-bypass'} & set(event.get('request_ids', []))
                       for event in events)
        evidence['final_empty'] = continuous._empty(server)
        assert server['child'].poll() is None
    evidence['status'] = 'passed'
    (tmp_path / 'held-cancellation-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
