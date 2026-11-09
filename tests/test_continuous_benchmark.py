"""Offered-arrival accounting must not hide rejections or dispatch delay."""

import json
import subprocess
import sys

import pytest

from benchmarks.continuous import _cases, arrival_offsets, case_indices, queue_delays, replay, summarize


def test_arrival_offsets_are_replayable_and_response_independent():
    assert arrival_offsets(4, 'sparse', sparse_interval=.5) == [0., .5, 1., 1.5]
    assert arrival_offsets(4, 'saturated', saturated_interval=.01) == [0., .01, .02, .03]
    assert arrival_offsets(5, 'bursty', burst_size=2, burst_interval=.5) == [0., 0., .5, .5, 1.]
    with pytest.raises(ValueError):
        arrival_offsets(0, 'sparse')
    with pytest.raises(ValueError):
        arrival_offsets(1, 'unknown')


def test_workloads_include_homogeneous_saturation_and_interleaved_bursts():
    assert case_indices(5, 'sparse') == [0, 1, 2, 3, 0]
    assert case_indices(5, 'saturated') == [3] * 5
    assert case_indices(9, 'bursty') == [3, 0, 3, 1, 2, 1, 2, 0, 3]


def test_rejection_rate_and_series_throughput_stay_visible_beside_latency():
    samples = [
        {'status': 'passed', 'series': 2, 'latency_s': .1, 'sojourn_s': .15, 'dispatch_lag_s': .05},
        {'status': 'passed', 'series': 1, 'latency_s': .3, 'sojourn_s': .3, 'dispatch_lag_s': 0.},
        {'status': 'rejected', 'series': 8, 'dispatch_lag_s': .01},
        {'status': 'failed', 'series': 3, 'dispatch_lag_s': .02},
    ]
    result = summarize(samples, 2.)
    assert (result['offered'], result['completed'], result['rejected'], result['failed']) == (4, 2, 1, 1)
    assert result['request_per_s'] == 1. and result['series_per_s'] == 1.5
    assert result['rejection_rate'] == .25 and result['p95_ms'] > result['p50_ms']
    assert result['offered_sojourn_p95_ms'] >= result['p95_ms']


def test_queue_delay_requires_explicit_correlated_server_events(tmp_path):
    trace = tmp_path / 'steps.jsonl'
    trace.write_text('\n'.join(json.dumps(event) for event in [
        {'event': 'enqueue', 'request_ids': ['pooling-a:uuid'], 'timestamp_ns': 1_000_000},
        {'event': 'schedule', 'wire_request_ids': ['pooling-a:uuid'], 'timestamp_ns': 4_000_000},
        {'event': 'schedule', 'wire_request_ids': ['pooling-a:uuid'], 'timestamp_ns': 8_000_000},
    ]))
    result = queue_delays(trace, [{'request_id': 'a'}, {'request_id': 'missing'}])
    assert result['samples'] == [{'request_id': 'a', 'queue_ms': 3.}]
    assert result['mean_ms'] == result['p95_ms'] == 3.
    assert queue_delays(tmp_path / 'absent', [])['mean_ms'] is None


def test_replay_keeps_errors_and_uses_client_identity():
    arrivals = [{'offset_s': 0., 'case': 'one', 'series': 1, 'encoded': b'{}'}] * 3

    def send(identity, encoded):
        index = int(identity.rsplit('-', 1)[1])
        if index == 1:
            return 503, {'error': 'queue full'}
        if index == 2:
            raise RuntimeError('network failed')
        return 200, {'request_id': f'pooling-{identity}', 'data': {'predictions': [{}]}}

    result = replay('unused', arrivals, model='test', request_prefix='replay', send=send)
    assert (result['completed'], result['rejected'], result['failed']) == (1, 1, 1)
    assert [row['request_id'] for row in result['samples']] == ['replay-0', 'replay-1', 'replay-2']
    assert all(row['latency_s'] >= 0 and row['dispatch_lag_s'] >= 0 for row in result['samples'])


def test_benchmark_help_does_not_import_gpu_packages():
    subprocess.run([sys.executable, '-c',
        'import sys; import benchmarks.continuous; '
        'assert not {"torch","vllm","timesfm"} & sys.modules.keys()'], check=True)


def test_interrupted_replay_cancels_owned_service_and_keeps_partial_evidence(tmp_path):
    calls = []

    def send(*args):
        raise KeyboardInterrupt('test interrupt')

    with pytest.raises(KeyboardInterrupt):
        replay('unused', [{'offset_s': 0., 'case': 'one', 'series': 1, 'encoded': b'{}'}],
               model='test', request_prefix='interrupt', send=send,
               cancel=lambda: calls.append('owned service stopped'), record_path=tmp_path / 'partial.json')
    assert calls == ['owned service stopped']
    evidence = json.loads((tmp_path / 'partial.json').read_text())
    assert evidence['status'] == 'interrupted' and len(evidence['samples']) == 1


def test_invalid_arrival_order_fails_before_starting_clients():
    rows = [{'offset_s': value, 'case': 'one', 'series': 1, 'encoded': b'{}'} for value in (.2, .1)]
    with pytest.raises(ValueError, match='chronological'):
        replay('unused', rows, model='test', request_prefix='invalid')


def test_performance_histories_do_not_require_observed_future_labels(monkeypatch):
    import numpy as np
    import benchmarks.model_evaluation as evaluation

    series = np.arange(4000, dtype=np.float32).reshape(2, 2000)
    observed = np.ones(2000, dtype=bool)
    observed[900:] = False
    monkeypatch.setattr(evaluation, '_assetops_data', lambda _: (
        series, observed, [f't{index}' for index in range(2000)], {'source': 'fixture'}))
    cases = _cases('unused', 4)
    assert [case['data']['parameters']['prediction_length'] for case in cases] == [128, 513, 129, 1024]
    assert len(cases[-1]['data']['inputs'][0]['target']) == 512
    assert all(case['provenance']['observed_future_labels_scored'] is False for case in cases)
    assert all(900 in case['provenance']['past_imputed_positions'] for case in cases)
