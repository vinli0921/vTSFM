"""CPU regressions for concurrency-aware throughput and failed-point reporting."""

import threading
import time
import json

import pytest

from benchmarks.model_performance import measure_requests


def test_concurrent_throughput_uses_elapsed_wall_time_and_preserves_samples():
    active = 0
    peak = 0
    lock = threading.Lock()
    def request(encoded):
        nonlocal active, peak
        assert encoded == b'preencoded'
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {'data': {'predictions': [{}, {}]}}
    result = measure_requests('unused', b'preencoded', 2, requests=8, clients=2, warmups=2, request=request)
    assert result['status'] == 'passed' and peak == 2
    assert len(result['samples']) == result['completed'] == 8
    assert result['request_per_s'] == pytest.approx(8 / result['elapsed_wall_s'])
    assert result['series_per_s'] == pytest.approx(16 / result['elapsed_wall_s'])
    assert result['p95_ms'] >= result['p50_ms'] > 0
    assert result['smoke_only']


def test_failed_requests_invalidate_throughput_and_cancel_queued_work():
    def request(encoded):
        raise RuntimeError('deliberate HTTP failure')
    result = measure_requests('unused', b'{}', 1, requests=10, clients=1, warmups=0, request=request)
    assert result['status'] == 'failed' and result['failed'] == 1
    assert result['cancelled'] == 9
    assert result['request_per_s'] is None and result['series_per_s'] is None
    assert 'deliberate HTTP failure' in result['samples'][0]['error']


def test_warmup_failure_prevents_a_measured_point():
    with pytest.raises(RuntimeError, match='warmup failed'):
        measure_requests('unused', b'{}', 1, requests=2, clients=1, warmups=1,
                         request=lambda _: {'data': {'predictions': []}})


def test_interruption_releases_workers_and_preserves_completed_samples(tmp_path):
    stopped = threading.Event()
    active = 0
    lock = threading.Lock()
    def request(encoded):
        nonlocal active
        with lock:
            active += 1
            index = active
        if index == 1:
            return {'data': {'predictions': [{}]}}
        if index == 2:
            raise KeyboardInterrupt()
        assert stopped.wait(2), 'Cancellation must release in-flight requests before draining'
        return {'data': {'predictions': [{}]}}
    path = tmp_path / 'partial.json'
    with pytest.raises(KeyboardInterrupt):
        measure_requests('unused', b'{}', 1, requests=8, clients=3, warmups=0,
                         request=request, cancel=stopped.set, record_path=path)
    result = json.loads(path.read_text())
    assert stopped.is_set() and result['interrupted']
    assert result['request_per_s'] is None
    assert any(sample['status'] == 'passed' for sample in result['samples'])
    assert len(result['samples']) == 8


@pytest.mark.parametrize('change', [{'requests': 0}, {'clients': 0}, {'series': True},
                                   {'warmups': -1}, {'requests': 1, 'clients': 2}])
def test_invalid_measurement_settings_fail_before_requests(change):
    settings = {'series': 1, 'requests': 2, 'clients': 1, 'warmups': 0, **change}
    with pytest.raises(ValueError):
        measure_requests('unused', b'{}', **settings, request=lambda _: pytest.fail('no request expected'))


PAYLOAD = {'payload_form': 'b64', 'payload_bytes': 175785, 'payload_sha256': 'f' * 64}


def test_point_records_carry_payload_metadata_when_first_written(tmp_path):
    """#96 review: the points/ file already names the payload it timed."""
    path = tmp_path / 'point.json'
    result = measure_requests('unused', b'{}', 1, requests=2, clients=1, warmups=0,
                              request=lambda _: {'data': {'predictions': [{}]}},
                              record_path=path, metadata=PAYLOAD)
    saved = json.loads(path.read_text())
    assert {key: saved[key] for key in PAYLOAD} == PAYLOAD
    assert {key: result[key] for key in PAYLOAD} == PAYLOAD
    assert saved['status'] == 'passed'


def test_interrupted_point_record_keeps_payload_metadata(tmp_path):
    def request(_):
        raise KeyboardInterrupt()
    path = tmp_path / 'partial.json'
    with pytest.raises(KeyboardInterrupt):
        measure_requests('unused', b'{}', 1, requests=2, clients=1, warmups=0,
                         request=request, record_path=path, metadata=PAYLOAD)
    saved = json.loads(path.read_text())
    assert saved['interrupted'] and {key: saved[key] for key in PAYLOAD} == PAYLOAD


def test_metadata_cannot_replace_measured_fields():
    with pytest.raises(ValueError, match='p50_ms'):
        measure_requests('unused', b'{}', 1, requests=1, warmups=0, metadata={'p50_ms': 1.0},
                         request=lambda _: pytest.fail('no request expected'))
