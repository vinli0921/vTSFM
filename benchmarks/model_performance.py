"""Matched HTTP measurements for the manual model profile, not model-only timings."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

from .common import percentile


# Fields measure_requests writes itself; caller metadata may not overwrite them.
_RESULT_KEYS = frozenset({
    'status', 'error', 'cleanup_error', 'interrupted', 'requests', 'clients', 'series_per_request',
    'warmups', 'completed', 'failed', 'cancelled', 'elapsed_wall_s', 'request_per_s', 'series_per_s',
    'mean_ms', 'p50_ms', 'p95_ms', 'samples', 'warmed_resident_gpu_bytes', 'peak_resident_gpu_bytes',
    'nvml_samples', 'memory_method', 'timing_boundary', 'smoke_only'})


def measure_requests(url, encoded, series, *, requests=100, clients=1, warmups=5,
                     monitor=None, request=None, cancel=None, record_path=None, metadata=None):
    """Measure pre-encoded request -> decoded response; warmups precede sampling.

    Throughput uses completed requests / elapsed wall time under concurrency.
    Numerical parity is checked separately, outside these measurement regions.
    Any failure invalidates the point; queued requests stop after the first error.
    ``metadata`` (JSON fields such as the payload form and size) is part of the
    result and of every ``record_path`` write, including an interrupted point's.
    """
    metadata = dict(metadata or {})
    if any(type(x) is not int or x <= 0 for x in (requests, clients, series)):
        raise ValueError('requests, clients, and series must be positive integers')
    if type(warmups) is not int or warmups < 0:
        raise ValueError('warmups must be a nonnegative integer')
    if requests < clients:
        raise ValueError('requests must be at least the requested client concurrency')
    reserved = set(metadata) & _RESULT_KEYS
    if reserved:
        raise ValueError(f'metadata cannot replace measured fields: {sorted(reserved)}')
    local = threading.local()
    sessions = []
    lock = threading.Lock()
    aborted = threading.Event()

    def invoke(index):
        if aborted.is_set():
            return {'index': index, 'status': 'cancelled'}
        try:
            if request is None:
                if not hasattr(local, 'session'):
                    import requests as http
                    local.session = http.Session()
                    local.session.trust_env = False
                    with lock:
                        sessions.append(local.session)
                started = time.perf_counter()
                response = local.session.post(url + '/pooling', data=encoded,
                    headers={'Content-Type': 'application/json'}, timeout=60)
                response.raise_for_status()
                body = response.json()
            else:
                started = time.perf_counter()
                body = request(encoded)
            elapsed = time.perf_counter() - started
            if len(body['data']['predictions']) != series:
                raise ValueError('Measured response has the wrong series count')
            return {'index': index, 'status': 'passed', 'latency_s': elapsed}
        except Exception as error:
            aborted.set()
            return {'index': index, 'status': 'failed', 'error': repr(error)}

    pool = ThreadPoolExecutor(max_workers=clients)
    error = None
    cleanup_error = None
    futures = []
    samples = []
    measured_started = None
    elapsed = None
    try:
        for result in pool.map(invoke, range(warmups)):
            if result['status'] != 'passed':
                raise RuntimeError(f'HTTP warmup failed: {result}')
        from .run import Sampler
        sampler = Sampler(monitor) if monitor is not None else None
        warmed_memory = monitor.used() if monitor is not None else None
        with sampler if sampler is not None else nullcontext():
            measured_started = time.perf_counter()
            futures = [pool.submit(invoke, index) for index in range(requests)]
            for future in as_completed(futures):
                samples.append(future.result())
            elapsed = time.perf_counter() - measured_started
    except BaseException as exception:
        error = exception
        elapsed = time.perf_counter() - measured_started if measured_started is not None else None
        raise
    finally:
        aborted.set()
        try:
            if error is not None and cancel is not None:
                # Stop the owned service first, releasing blocked HTTP workers
                # before draining their threads and letting fixture teardown run.
                try:
                    cancel()
                except Exception as exception:
                    cleanup_error = repr(exception)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for session in sessions:
                session.close()
        if error is not None and record_path is not None:
            completed = []
            for index, future in enumerate(futures):
                if future.cancelled():
                    completed.append({'index': index, 'status': 'cancelled'})
                elif future.exception() is not None:
                    completed.append({'index': index, 'status': 'failed', 'error': repr(future.exception())})
                else:
                    completed.append(future.result())
            path = Path(record_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**metadata, 'status': 'failed', 'error': repr(error),
                'cleanup_error': cleanup_error, 'interrupted': isinstance(error, (KeyboardInterrupt, SystemExit)),
                'requests': requests, 'clients': clients, 'elapsed_wall_s': elapsed,
                'request_per_s': None, 'series_per_s': None, 'samples': completed}, indent=2) + '\n')
    latencies = [row['latency_s'] for row in samples if row['status'] == 'passed']
    failed = sum(row['status'] == 'failed' for row in samples)
    cancelled = sum(row['status'] == 'cancelled' for row in samples)
    valid = not failed and not cancelled and len(latencies) == requests
    result = {**metadata, 'status': 'passed' if valid else 'failed', 'requests': requests, 'clients': clients,
            'series_per_request': series, 'warmups': warmups, 'completed': len(latencies),
            'failed': failed, 'cancelled': cancelled, 'elapsed_wall_s': elapsed,
            'request_per_s': len(latencies) / elapsed if valid else None,
            'series_per_s': series * len(latencies) / elapsed if valid else None,
            'mean_ms': 1000 * sum(latencies) / len(latencies) if latencies else None,
            'p50_ms': 1000 * percentile(latencies, 0.5) if latencies else None,
            'p95_ms': 1000 * percentile(latencies, 0.95) if latencies else None,
            'samples': sorted(samples, key=lambda row: row['index']),
            'warmed_resident_gpu_bytes': warmed_memory,
            'peak_resident_gpu_bytes': max(row[1] for row in sampler.samples) if sampler else None,
            'nvml_samples': sampler.samples if sampler else [],
            'memory_method': 'NVML whole-device residency sampled every 20ms; may miss short peaks',
            'timing_boundary': 'pre-encoded HTTP request through JSON response decode; startup and warmup excluded',
            'smoke_only': requests < 100}
    if record_path is not None:
        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
