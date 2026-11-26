"""Compare TimesFM dense eager and CUDA Graph replay over HTTP for Chiller 6."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time
import traceback

from benchmarks.common import percentile
from benchmarks.timesfm_graph_http import (
    OWNERSHIP_METHOD, _GuardedMonitor, _StartupSession, _reference, _route_delta, _route_state, _rpc,
)
from benchmarks.timesfm_native_kv import _sha, _source_metadata, _write


ROOT = Path(__file__).resolve().parents[1]
ORDERS = (('dense-eager', 'dense-graph'), ('dense-graph', 'dense-eager'))
GEOMETRY = (5, 512, 96)
LEVELS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
RUN_LABELS = ('A', 'B-values', 'C-mask', 'A-repeat', 'shape-fallback', 'A-after-fallback')
GRAPH_MODE_ENV = 'VTSFM_TIMESFM_FORECAST_MODE'
GRAPH_SHAPE_ENV = 'VTSFM_TIMESFM_FORECAST_SHAPE'
FALLBACK_LABELS = ('C-mask', 'shape-fallback')


def _with_ids(case, label):
    payload = deepcopy(case)
    for row, item in enumerate(payload['data']['inputs']):
        item['item_id'] = f'{label}-{row}'
    return payload


def _value_and_mask_variants(case):
    inputs = case['data']['inputs']
    changed_row = min(1, len(inputs) - 1)
    target = inputs[changed_row]['target']
    # Shorten one history without changing the width of the padded batch.
    # TimesFM represents its left padding as NaNs and uses the eager fallback.
    crop = min(48, len(target) - 1)
    if crop < 1:
        raise ValueError('The selected Chiller fixture cannot vary its effective observation mask')
    values = _with_ids(case, 'B-values')
    masked = _with_ids(case, 'C-mask')
    values['name'] += '-changed-values'
    masked['name'] += '-shorter-history'
    values['data']['inputs'][changed_row]['target'] = [-0.75 * v + 0.23 for v in target]
    masked['data']['inputs'][changed_row]['target'] = target[crop:]
    return values, masked


def _cases(args):
    import numpy as np
    from model_integration_support import validate_cases

    document = json.loads(args.cases.read_bytes())
    candidates = document['cases'] if isinstance(document, dict) else document
    original, = validate_cases('timesfm25', [case for case in candidates if args.case_filter in case['name']])
    saved, = [entry for entry in json.loads(args.archived_native.read_bytes())['cases']
              if entry['case']['name'] == original['name']]
    if saved['case']['data'] != original['data']:
        raise ValueError('The selected archived payload differs from the serving manifest')
    if _shape(original) != GEOMETRY:
        raise ValueError(f'The selected fixture must have geometry {GEOMETRY}')
    if original['data']['parameters']['quantile_levels'] != LEVELS:
        raise ValueError('The selected fixture must request all nine native quantiles')
    case_b, case_c = _value_and_mask_variants(original)
    if _shape(case_c) != GEOMETRY:
        raise ValueError('The ragged variant must retain the original padded geometry')
    fallback = _with_ids(original, 'shape-fallback')
    fallback['name'] += '-shape-fallback'
    settings = fallback['data']['parameters']
    settings['prediction_length'] = 33 if settings['prediction_length'] != 33 else 17
    return validate_cases('timesfm25', [original, case_b, case_c, fallback]), saved, {
        'manifest_sha256': _sha(args.cases),
        'archived_native_sha256': _sha(args.archived_native),
        'archived_case_native_forecast_sha256': hashlib.sha256(
            np.asarray(saved['expected'], dtype=np.float32).tobytes()).hexdigest(),
        'source': original.get('provenance'),
        'case': original['name'],
        'alternate_inputs': 'B modifies one history by -0.75*x + 0.23; C crops a history on the left. '
                            'Variants are parity diagnostics rather than observed-label accuracy cases.',
    }


def _shape(case):
    data = case['data']
    return (len(data['inputs']), max(len(row['target']) for row in data['inputs']),
            data['parameters']['prediction_length'])


def _verify_reference(native, saved):
    import numpy as np

    original, changed, masked, fallback = native['cases']
    expected = np.asarray(saved['expected'], dtype=np.float32)
    actual = np.asarray(original['expected'], dtype=np.float32)
    if actual.tobytes() != expected.tobytes() or actual.shape != expected.shape:
        raise AssertionError('Fresh independent native forecast differs from archived FP32 values')
    for case in (changed, masked):
        other = np.asarray(case['expected'], dtype=np.float32)
        if other.shape != actual.shape or np.allclose(other, actual, rtol=1e-4, atol=1e-4):
            raise AssertionError(f'{case["case"]["name"]} does not distinguish replayed output from stale A')
    if _shape(fallback['case']) == _shape(original['case']):
        raise AssertionError('Shape fallback must differ from the captured geometry')
    return hashlib.sha256(actual.tobytes()).hexdigest()


def _comparison(body, case, expected, record):
    import numpy as np
    from model_integration_support import compare_response

    try:
        parity = compare_response(body, case, expected, rtol=1e-4, atol=1e-4)
    except AssertionError as error:
        record.update(status='error', error=str(error))
        predictions = body.get('data', {}).get('predictions', [])
        if (len(predictions) == len(expected)
                and all(isinstance(item, dict) and all(str(level) in item for level in LEVELS)
                        for item in predictions)):
            try:
                observed = np.asarray([[item[str(level)] for level in LEVELS]
                                      for item in predictions], dtype=np.float64)
            except (TypeError, ValueError):
                raise error from None
            reference = np.asarray(expected, dtype=np.float64)
            if observed.shape == reference.shape and np.isfinite(observed).all():
                delta = np.abs(observed - reference)
                record.update(compared_values=reference.size, max_abs_error=float(delta.max()),
                              outside_tolerance=int((delta > 1e-4 + 1e-4 * np.abs(reference)).sum()))
        raise
    actual = np.asarray([[item[str(level)] for level in LEVELS]
                         for item in body['data']['predictions']], dtype=np.float32)
    reference = np.asarray(expected, dtype=np.float32)
    result = hashlib.sha256(actual.tobytes()).hexdigest()
    oracle = hashlib.sha256(reference.tobytes()).hexdigest()
    record.update(parity, status='ok', served_forecast_sha256=result, native_forecast_sha256=oracle,
                  bit_exact_to_native=result == oracle)


def _server(args, mode, shape):
    from vtsfm.cli import build_vllm_command

    command, environment = build_vllm_command(adapter='timesfm25', port=args.port, benchmark=True,
                                              max_num_seqs=1, forecast_precision='fp32',
                                              forecast_execution='eager',
                                              timesfm_forecast_mode=mode, timesfm_forecast_shape=shape)
    if '--no-async-scheduling' in command:
        command.remove('--no-async-scheduling')
    command += ['--no-async-scheduling', '--gpu-memory-utilization', str(args.gpu_memory_utilization)]
    environment.update(_environment(args))
    return command, environment


def _environment(args):
    return dict(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                NVIDIA_TF32_OVERRIDE='0', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4',
                PYTHONUNBUFFERED='1', VTSFM_PRECISION='fp32', VTSFM_TEST_PRECISION='fp32',
                VTSFM_FORECAST_EXECUTION='eager',
                PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]))


def _route(session, url):
    return _rpc(session, url, 'bench_timesfm_forecast')


def _point(args, owner, monitor, mode, repetition, native):
    import requests
    from benchmarks.model_performance import measure_requests
    from benchmarks.run import ensure_port_free, ensure_worker_gpu, memory, wait_ready

    case_a, case_b, case_c, fallback = native['cases']
    rows, width, horizon = _shape(case_a['case'])
    directory = args.output / f'repetition-{repetition}-{mode}'
    directory.mkdir()
    record = {'mode': mode, 'repetition': repetition, 'geometry': [rows, width, horizon],
              'status': 'running', 'parity': {}, 'routes': {}, 'route_deltas': {}}
    server = guarded = None
    stopped = False

    def stop_server():
        nonlocal stopped
        if server is not None and not stopped:
            owner.stop(server)
            stopped = True

    def encode(case):
        return json.dumps({'model': native['model_id'], 'task': 'plugin', 'data': case['data']},
                          separators=(',', ':'), allow_nan=False).encode()

    request_ids, clients = [], []
    request_lock = threading.Lock()
    request_count = 0
    thread_client = threading.local()
    try:
        command, environment = _server(args, mode, (rows, width, horizon))
        record['command'] = command
        record['worker_environment'] = {key: environment.get(key) for key in (
            GRAPH_MODE_ENV, GRAPH_SHAPE_ENV, 'VTSFM_FORECAST_EXECUTION',
            'VTSFM_PRECISION', 'NVIDIA_TF32_OVERRIDE')}
        url = f'http://127.0.0.1:{args.port}'
        monitor.assert_idle()
        ensure_port_free('127.0.0.1', args.port)
        with requests.Session() as control:
            control.trust_env = False
            start = time.perf_counter()
            server = owner.spawn(command, environment, directory.name)
            guarded = _GuardedMonitor(monitor, server, record)
            wait_ready(server, url, _StartupSession(control, guarded), timeout=args.startup_timeout)
            record['startup_ready_s'] = time.perf_counter() - start
            record['worker_config'] = memory(control, url, 'vllm', 'config')
            ensure_worker_gpu(record['worker_config'], monitor.uuid)
            record['startup_worker_memory'] = memory(control, url, 'vllm', 'stats')
            record['startup_sampled_device_bytes'] = guarded.used()
            ready = _route(control, url)
            record['routes']['ready'] = ready
            _route_state(ready, mode, ready=True)
            if ready.get('shape') != list(GEOMETRY):
                raise AssertionError(f'The TimesFM owner did not use {GEOMETRY}: {ready}')

            def send(encoded):
                nonlocal request_count
                if guarded.failure:
                    raise RuntimeError(guarded.failure)
                with request_lock:
                    identity = f'timesfm-chiller-http-r{repetition}-{mode}-{request_count}'
                    request_count += 1
                    if not hasattr(thread_client, 'session'):
                        thread_client.session = requests.Session()
                        thread_client.session.trust_env = False
                        clients.append(thread_client.session)
                    client = thread_client.session
                response = client.post(url + '/pooling', data=encoded,
                    headers={'Content-Type': 'application/json', 'X-Request-Id': identity},
                    timeout=args.request_timeout)
                response.raise_for_status()
                body = response.json()
                if body.get('request_id') != f'pooling-{identity}':
                    raise AssertionError('HTTP response request identity was stale or mismatched')
                with request_lock:
                    request_ids.append(body['request_id'])
                return body

            def checked(label, entry, *, change_ids=False):
                case = _with_ids(entry['case'], label) if change_ids else entry['case']
                request = encode(case)
                guarded.assert_owned()
                started = time.perf_counter()
                body = send(request)
                elapsed = time.perf_counter() - started
                guarded.assert_owned()
                _write(directory / f'{label}-request.json', json.loads(request))
                _write(directory / f'{label}-response.json', body)
                proof = record['parity'][label] = {'elapsed_wall_s': elapsed}
                _comparison(body, case, entry['expected'], proof)
                record['routes'][label] = _route(control, url)

            entries = (case_a, case_b, case_c, case_a, fallback, case_a)
            before = ready
            for label, entry in zip(RUN_LABELS, entries, strict=True):
                checked(label, entry, change_ids=label in ('A-repeat', 'A-after-fallback'))
                after = record['routes'][label]
                fallback_expected = label in FALLBACK_LABELS
                record['route_deltas'][label] = _route_delta(before, after, mode,
                    calls=int(not fallback_expected), fallbacks=int(fallback_expected))
                before = after
            data = encode(case_a['case'])
            record['timed_payload_sha256'] = hashlib.sha256(data).hexdigest()
            record['measurements'] = {}
            order = args.clients if repetition == 1 else args.clients[::-1]
            for concurrency in order:
                key = str(concurrency)
                arm = record['measurements'][key] = {'clients': concurrency}
                warmups = max(concurrency, args.warmups)
                arm['route_before_warmup'] = _route(control, url)
                arm['worker_memory_before_warmup'] = memory(control, url, 'vllm', 'reset')
                guarded.assert_owned()
                _write(directory / 'point.json', record)
                measured = measure_requests(url, data, rows, requests=args.requests, clients=concurrency,
                    warmups=warmups, monitor=guarded, request=send, cancel=stop_server,
                    record_path=directory / f'requests-c{concurrency}.json')
                if measured['status'] != 'passed' or measured['completed'] != args.requests:
                    raise RuntimeError('An HTTP request failed; point is invalid')
                arm['worker_memory_after_timing'] = memory(control, url, 'vllm', 'stats')
                arm['route_after_timing'] = _route(control, url)
                arm['route_delta_warmup_and_timed'] = _route_delta(arm['route_before_warmup'],
                    arm['route_after_timing'], mode, calls=args.requests + warmups)
                arm['measurement'] = {name: value for name, value in measured.items()
                                      if name not in ('samples', 'nvml_samples')}
                checked(f'A-after-timing-c{concurrency}', case_a, change_ids=True)
            guarded.assert_owned()
            record['successful_http_request_ids'] = request_ids
            record['status'] = 'ok'
    except BaseException as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        try:
            for client in clients:
                client.close()
            stop_server()
            record['owned_server_stopped'] = stopped or server is None
            if server is not None and guarded is not None and not record['gpu_ownership']['foreign_pids']:
                monitor.wait_idle()
                record['gpu_idle_after_cleanup'] = True
        except Exception as error:
            record.update(status='error', cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            _write(directory / 'point.json', record)
    return record


def _timings(points, output, concurrency):
    summary = {str(clients): {} for clients in concurrency}
    if len({point['timed_payload_sha256'] for point in points}) != 1:
        raise AssertionError('Eager and graph timings must use identical requests')
    for clients in concurrency:
        for mode in ORDERS[0]:
            group = [point for point in points if point['mode'] == mode]
            measured = [json.loads((output / f'repetition-{p["repetition"]}-{mode}' /
                                   f'requests-c{clients}.json').read_bytes()) for p in group]
            latencies = [s['latency_s'] for run in measured for s in run['samples']]
            elapsed = sum(run['elapsed_wall_s'] for run in measured)
            summary[str(clients)][mode] = {'completed': len(latencies), 'elapsed_wall_s': elapsed,
                'request_per_s': len(latencies) / elapsed, 'p50_ms': 1e3 * percentile(latencies, .5),
                'p95_ms': 1e3 * percentile(latencies, .95),
                'peak_resident_gpu_bytes': max(run['peak_resident_gpu_bytes'] for run in measured)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--archived-native', type=Path, required=True)
    parser.add_argument('--case-filter', default='condenser_water_flow')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--requests', type=int, default=100, help='Requests per process and order')
    parser.add_argument('--clients', nargs='+', type=int, choices=(1, 4), default=(1, 4))
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--port', type=int, default=18579)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--startup-timeout', type=float, default=240)
    parser.add_argument('--request-timeout', type=float, default=60)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.35)
    args = parser.parse_args()
    if (args.requests < max(args.clients) or len(set(args.clients)) != len(args.clients)
            or args.warmups < 0 or args.gpu < 0 or not 1 <= args.port <= 65535
            or not 0 < args.gpu_memory_utilization < 1
            or any(not math.isfinite(t) or t <= 0 for t in (args.startup_timeout, args.request_timeout))):
        parser.error('Invalid counts, GPU, port, utilization or timeout')
    if platform.system() != 'Linux':
        parser.error('HTTP GPU timing requires Linux with the pinned TimesFM environment')
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'tests'))
    from benchmarks.run import GpuMonitor, OwnedProcesses

    record = dict(_source_metadata(), probe_sha256=_sha(Path(__file__)), status='running',
                  orders=ORDERS, ownership_method=OWNERSHIP_METHOD, points=[],
                  timing_boundary='One or four keep-alive clients, one vLLM slot, through response JSON decode. '
                                  'Cold start, warmups, parity and RPCs are outside timing.',
                  smoke_only=args.requests < 100)
    monitor = None
    default = signal.getsignal(signal.SIGTERM)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        cases, archived, provenance = _cases(args)
        record['provenance'] = provenance
        monitor = GpuMonitor(args.gpu)
        monitor.assert_idle()
        record['gpu'] = monitor.metadata()
        _write(args.output / 'manifest.json', record)
        with OwnedProcesses(args.output) as owner:
            native = _reference(args, owner, monitor, cases, dict(os.environ, **_environment(args),
                                VLLM_PLUGINS=''))
            record['archived_native_reproduced_sha256'] = _verify_reference(native, archived)
            for repetition, order in enumerate(ORDERS, 1):
                for mode in order:
                    point = _point(args, owner, monitor, mode, repetition, native)
                    record['points'].append(point)
                    _write(args.output / 'manifest.json', record)
                    print(json.dumps({'mode': mode, 'repetition': repetition,
                                      'measurements': point['measurements']}), flush=True)
        record['bit_exact_across_arms'] = {
            label: len({point['parity'][label]['served_forecast_sha256'] for point in record['points']}) == 1
            for label in record['points'][0]['parity']
        }
        record['timings'] = _timings(record['points'], args.output, args.clients)
        record['status'] = 'ok'
        return 0
    except BaseException as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        signal.signal(signal.SIGTERM, default)
        try:
            if monitor is not None:
                monitor.close()
        finally:
            _write(args.output / 'manifest.json', record)


if __name__ == '__main__':
    sys.exit(main())
