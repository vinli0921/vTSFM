"""Replay offered HTTP arrivals for whole-forecast and block serving.

Help, trace construction and metric aggregation are CPU-safe. GPU/model loading
occurs only when the explicit benchmark CLI starts its owned servers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .common import percentile


ROOT = Path(__file__).resolve().parents[1]


def arrival_offsets(count, scenario, *, sparse_interval=1.0, saturated_interval=0.001,
                    burst_size=16, burst_interval=0.25):
    """Absolute offered times; arrival generation never waits for responses."""
    if type(count) is not int or count < 1 or type(burst_size) is not int or burst_size < 1:
        raise ValueError('count and burst_size must be positive integers')
    if any(not math.isfinite(value) or value < 0 for value in
           (sparse_interval, saturated_interval, burst_interval)):
        raise ValueError('Arrival intervals must be finite and nonnegative')
    if scenario == 'sparse':
        return [index * sparse_interval for index in range(count)]
    if scenario == 'saturated':
        return [index * saturated_interval for index in range(count)]
    if scenario == 'bursty':
        return [(index // burst_size) * burst_interval for index in range(count)]
    raise ValueError(f'Unknown arrival scenario: {scenario}')


def case_indices(count, scenario, case_count=4):
    """Include homogeneous saturation and fixed mixed arrivals/departures.

    The burst trace interleaves long/short forecasts and context sizes before
    any measurement is taken; the exact assignments are saved in the manifest.
    """
    if count < 1 or case_count < 1:
        raise ValueError('count and case_count must be positive')
    if scenario == 'sparse':
        return [index % case_count for index in range(count)]
    if scenario == 'saturated':
        return [case_count - 1] * count
    if scenario == 'bursty':
        pattern = (3, 0, 3, 1, 2, 1, 2, 0)
        return [pattern[index % len(pattern)] % case_count for index in range(count)]
    raise ValueError(f'Unknown arrival scenario: {scenario}')


def summarize(samples, elapsed):
    """Keep rejection/error counts beside latency; report accepted throughput."""
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise ValueError('elapsed must be finite and positive')
    completed = [sample for sample in samples if sample['status'] == 'passed']
    rejected = [sample for sample in samples if sample['status'] == 'rejected']
    errors = [sample for sample in samples if sample['status'] == 'failed']
    latency = [sample['latency_s'] for sample in completed]
    sojourn = [sample['sojourn_s'] for sample in completed]
    return {
        'offered': len(samples), 'completed': len(completed), 'rejected': len(rejected),
        'failed': len(errors), 'rejection_rate': len(rejected) / len(samples) if samples else 0.,
        'elapsed_wall_s': elapsed, 'request_per_s': len(completed) / elapsed,
        'series_per_s': sum(sample['series'] for sample in completed) / elapsed,
        'p50_ms': 1000 * percentile(latency, .5) if latency else None,
        'p95_ms': 1000 * percentile(latency, .95) if latency else None,
        'offered_sojourn_p95_ms': 1000 * percentile(sojourn, .95) if sojourn else None,
        'dispatch_lag_p95_ms': 1000 * percentile([sample['dispatch_lag_s'] for sample in samples], .95)
        if samples else None,
    }


def _canonical(identity):
    # The production wire ID is '<external pooling id>:<invocation UUID>'.
    return identity.rsplit(':', 1)[0]


def queue_delays(path, samples):
    """Correlate explicit admission/scheduling timestamps, never infer GPU queue time.

    Trace events can identify one request through request_id or multiple through
    request_ids. A wire_request_ids list takes precedence over internal IDs.
    """
    entered, first = {}, {}
    if not Path(path).exists():
        return {'method': 'unavailable: no scheduler trace', 'samples': [], 'mean_ms': None,
                'p50_ms': None, 'p95_ms': None}
    for line in Path(path).read_text().splitlines():
        event = json.loads(line)
        kind = event.get('event')
        if kind not in ('enqueue', 'schedule'):
            continue
        stamp = event.get('timestamp_ns')
        if type(stamp) is not int:
            continue
        identities = event.get('wire_request_ids') or event.get('request_ids') or [event.get('request_id')]
        destination = entered if kind == 'enqueue' else first
        for identity in identities:
            if isinstance(identity, str):
                destination.setdefault(_canonical(identity), stamp)
    values = []
    for sample in samples:
        identity = f"pooling-{sample['request_id']}"
        if identity in entered and identity in first and first[identity] >= entered[identity]:
            values.append({'request_id': sample['request_id'],
                           'queue_ms': (first[identity] - entered[identity]) / 1e6})
    delays = [value['queue_ms'] for value in values]
    return {'method': 'engine enqueue to first scheduled block' if delays else
            'unavailable: no correlated enqueue/schedule timestamps',
            'samples': values, 'mean_ms': sum(delays) / len(delays) if delays else None,
            'p50_ms': percentile(delays, .5) if delays else None,
            'p95_ms': percentile(delays, .95) if delays else None}


def replay(url, arrivals, *, model, request_prefix, monitor=None, send=None, timeout=120,
           cancel=None, record_path=None):
    """Replay pre-encoded bodies; one worker per arrival prevents closed-loop throttling.

    Latency starts immediately before HTTP submission and ends after JSON decode.
    Offered sojourn additionally includes dispatch lateness, recorded separately.
    Errors remain in raw samples. No success-dependent cancellation hides tail work.
    """
    if not arrivals:
        raise ValueError('At least one arrival is required')
    offsets = [arrival['offset_s'] for arrival in arrivals]
    if any(not math.isfinite(offset) or offset < 0 for offset in offsets) or offsets != sorted(offsets):
        raise ValueError('Arrival offsets must be finite, nonnegative and chronological')
    from .run import Sampler

    origin = time.perf_counter()
    samples = []

    def invoke(index, arrival):
        identity = f'{request_prefix}-{index}'
        started = time.perf_counter()
        sample = {'index': index, 'request_id': identity, 'case': arrival['case'],
                  'series': arrival['series'], 'offset_s': arrival['offset_s'],
                  'dispatch_lag_s': max(0., started - origin - arrival['offset_s']),
                  'submitted_s': started - origin}
        try:
            if send is None:
                import requests
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(url + '/pooling', data=arrival['encoded'],
                        headers={'Content-Type': 'application/json', 'X-Request-Id': identity},
                        timeout=timeout)
                    code = response.status_code
                    try:
                        body = response.json()
                    except ValueError:
                        body = {'error': response.text}
            else:
                code, body = send(identity, arrival['encoded'])
            sample['http_status'] = code
            if code in (429, 503):
                sample.update(status='rejected', error=body)
            elif code != 200:
                sample.update(status='failed', error=body)
            elif (body.get('request_id') != f'pooling-{identity}'
                    or len(body.get('data', {}).get('predictions', [])) != arrival['series']):
                sample.update(status='failed', error='Invalid forecast response identity/series count')
            else:
                sample['status'] = 'passed'
        except Exception as error:
            sample.update(status='failed', error=repr(error))
        ended = time.perf_counter()
        sample.update(latency_s=ended - started, sojourn_s=ended - origin - arrival['offset_s'],
                      completed_s=ended - origin)
        return sample

    sampler = Sampler(monitor, interval=.02) if monitor is not None else None
    clients = ThreadPoolExecutor(max_workers=len(arrivals))
    futures = []
    interrupted = None
    try:
        with sampler if sampler is not None else nullcontext():
            for index, arrival in enumerate(arrivals):
                remaining = origin + arrival['offset_s'] - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                futures.append(clients.submit(invoke, index, arrival))
            for future in as_completed(futures):
                samples.append(future.result())
    except BaseException as error:
        interrupted = error
        # Stop the owned service before draining clients blocked in its HTTP call.
        if cancel is not None:
            cancel()
        raise
    finally:
        clients.shutdown(wait=True, cancel_futures=True)
        if interrupted is not None and record_path is not None:
            rows = []
            for index, future in enumerate(futures):
                try:
                    rows.append(future.result())
                except BaseException as error:
                    rows.append({'index': index, 'status': 'failed', 'error': repr(error)})
            path = Path(record_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                'status': 'interrupted', 'error': repr(interrupted), 'samples': rows,
                'elapsed_wall_s': time.perf_counter() - origin,
            }, indent=2) + '\n')
    elapsed = time.perf_counter() - origin
    result = {**summarize(samples, elapsed),
              'samples': sorted(samples, key=lambda value: value['index']),
              'peak_resident_gpu_bytes': max(value[1] for value in sampler.samples) if sampler else None,
              'nvml_samples': sampler.samples if sampler else [],
              'timing_boundary': 'pre-encoded HTTP submit through JSON decode; startup/warmup excluded',
              'load_method': 'absolute offered-arrival replay; clients never wait for prior completions',
              'smoke_only': len(arrivals) < 100}
    return result


def _queue_counter(url):
    """Optional built-in metric fallback for mean server queue delay."""
    import requests
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(url + '/metrics', timeout=10)
        response.raise_for_status()
    totals = {}
    for field in ('sum', 'count'):
        pattern = re.compile(r'^vllm:request_queue_time_seconds_' + field + r'(?:\{[^}]*\})?\s+([0-9.eE+-]+)$')
        values = [float(match.group(1)) for line in response.text.splitlines()
                  if (match := pattern.match(line))]
        if values:
            totals[field] = sum(values)
    for field in ('running', 'waiting'):
        pattern = re.compile(r'^vllm:num_requests_' + field + r'(?:\{[^}]*\})?\s+([0-9.eE+-]+)$')
        values = [float(match.group(1)) for line in response.text.splitlines()
                  if (match := pattern.match(line))]
        if values:
            totals[field] = sum(values)
    return totals


def _allocator_stats(url, *, reset=False):
    import requests
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(url + '/collective_rpc', json={
            'method': 'bench_memory_reset' if reset else 'bench_memory_stats',
            'args': [], 'timeout': 20}, timeout=25)
        response.raise_for_status()
    result = response.json()['results']
    if len(result) != 1:
        raise ValueError('Memory accounting expects exactly one GPU worker')
    return result[0]


def _drain_requests(url, path, timeout=30.):
    """Record an empty engine and worker before primary counters/allocator peaks."""
    import requests

    began = time.monotonic()
    record = {'status': 'running', 'observations': []}
    try:
        with requests.Session() as session:
            session.trust_env = False
            while True:
                engine = _queue_counter(url)
                response = session.post(url + '/collective_rpc', json={
                    'method': 'bench_request_state', 'args': [], 'timeout': 20}, timeout=25)
                response.raise_for_status()
                states = response.json()['results']
                if len(states) != 1:
                    raise ValueError('Request state accounting expects exactly one GPU worker')
                state = states[0]
                record['observations'].append({'elapsed_s': time.monotonic() - began,
                                               'engine': engine, 'worker': state})
                if not {'running', 'waiting'} <= engine.keys():
                    raise RuntimeError('Required engine running/waiting gauges are unavailable')
                if not {'worker_requests', 'worker_slots'} <= state.keys():
                    raise RuntimeError('Required worker request/slot counts are unavailable')
                if engine['running'] == engine['waiting'] == 0 and not any(state.values()):
                    record['status'] = 'passed'
                    return record
                if time.monotonic() - began >= timeout:
                    raise TimeoutError('Warmup requests did not drain; keep state evidence')
                time.sleep(.05)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(record, indent=2) + '\n')


def _cases(data_dir, count):
    # This is a serving workload, not an observed-label accuracy task. The
    # fixture has no fully observed 513/1024-step future span; requiring one
    # would incorrectly prevent timing those supported forecasting horizons.
    from .model_evaluation import _assetops_data
    from .prepare_data import COLUMNS

    series, observed, timestamps, provenance = _assetops_data(data_dir)
    origin = series.shape[1] - 1024
    cases = []
    for index, (context, horizon, rows) in enumerate(((128, 128, 1), (128, 513, 2),
                                                   (512, 129, 1), (512, 1024, 2))):
        if origin < context:
            raise ValueError('The prepared fixture is too short for replay contexts')
        inputs = [{'item_id': f'mp_1/{COLUMNS[row][0]}/origin-{origin}/row-{row}',
                   'target': series[row, origin - context:origin].tolist()} for row in range(rows)]
        cases.append({'name': f'case-{index}-b{rows}-c{context}-h{horizon}',
                      'data': {'inputs': inputs, 'parameters': {
                          'prediction_length': horizon, 'quantile_levels': [q / 10 for q in range(1, 10)]}},
                      'provenance': {**provenance, 'origin': origin,
                          'history_start': timestamps[origin - context], 'forecast_start': timestamps[origin],
                          'channel_rows': list(range(rows)),
                          'past_imputed_positions': [position for position in range(origin - context, origin)
                                                     if not observed[position]],
                          'observed_future_labels_scored': False,
                          'repeated_queries': True,
                          'sampling_note': 'Repeated correlated channel/history windows from one compressor'}})
    return [cases[index % len(cases)] for index in range(count)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--warmup-requests', type=int, default=0,
                        help='Untimed offered-arrival warmup per mode/scenario; zero preserves the original protocol')
    parser.add_argument('--warmup-timeout', type=float, default=120.,
                        help='Timeout for untimed warmup requests, including lazy compilation')
    parser.add_argument('--modes', default='single,whole,continuous')
    parser.add_argument('--scenarios', default='sparse,saturated,bursty')
    parser.add_argument('--port', type=int, default=18515)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--queue-limit', type=int, default=128)
    parser.add_argument('--diagnostic-requests', type=int, default=24,
                        help='Separately replay this prefix with scheduler traces; zero disables it')
    parser.add_argument('--sparse-interval', type=float, default=1.)
    parser.add_argument('--saturated-interval', type=float, default=.001)
    parser.add_argument('--burst-size', type=int, default=16)
    parser.add_argument('--burst-interval', type=float, default=.25)
    args = parser.parse_args(argv)
    modes, scenarios = args.modes.split(','), args.scenarios.split(',')
    if not modes or set(modes) - {'single', 'whole', 'continuous'} or len(set(modes)) != len(modes):
        parser.error('--modes must contain unique single,whole,continuous values')
    if not scenarios or set(scenarios) - {'sparse', 'saturated', 'bursty'} or len(set(scenarios)) != len(scenarios):
        parser.error('--scenarios must contain unique sparse,saturated,bursty values')
    if args.queue_limit < 1:
        parser.error('--queue-limit must be positive')
    if args.requests < 1:
        parser.error('--requests must be positive')
    if args.warmup_requests < 0:
        parser.error('--warmup-requests must be nonnegative')
    if not math.isfinite(args.warmup_timeout) or args.warmup_timeout <= 0:
        parser.error('--warmup-timeout must be finite and positive')
    if args.diagnostic_requests < 0:
        parser.error('--diagnostic-requests must be nonnegative')
    args.output = args.output.resolve()
    args.data_dir = args.data_dir.resolve()
    cases = _cases(args.data_dir, 4)
    offsets_by_scenario = {scenario: arrival_offsets(args.requests, scenario,
        sparse_interval=args.sparse_interval, saturated_interval=args.saturated_interval,
        burst_size=args.burst_size, burst_interval=args.burst_interval) for scenario in scenarios}
    indices_by_scenario = {scenario: case_indices(args.requests, scenario) for scenario in scenarios}
    args.output.mkdir(parents=True, exist_ok=False)
    from .run import GpuMonitor, OwnedProcesses, ensure_port_free, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command
    import requests

    monitor = GpuMonitor(args.gpu)
    spec = get_adapter('timesfm25')
    metadata = {'model': spec.model_id, 'revision': spec.model_revision,
                'gpu': monitor.metadata(), 'packages': {name: importlib.metadata.version(name)
                    for name in ('torch', 'vllm', 'timesfm')},
                'cases': cases, 'modes': modes, 'scenarios': scenarios, 'queue_limit': args.queue_limit,
                'arrival_offsets_s': offsets_by_scenario,
                'arrival_case_indices': indices_by_scenario,
                'input_kind': 'AssetOpsBench sensor histories with documented causal fills; repeated queries; no label scoring',
                'settings': (f"{os.environ.get('VTSFM_FORECAST_EXECUTION', 'eager')} FP32, "
                             'TF32 disabled, async scheduling disabled'),
                'warmup_requests_per_scenario': args.warmup_requests,
                'warmup_timeout_s': args.warmup_timeout,
                'parity_reference_mode': modes[0], 'parity_tolerance': {'rtol': 1e-4, 'atol': 1e-4},
                'diagnostic_prefix_requests': min(args.requests, args.diagnostic_requests),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_working_changes': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT / 'src/vtsfm').rglob('*.py'))},
                'results': [], 'commands': {}}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest_hash = hashlib.sha256((args.output / 'manifest.json').read_bytes()).hexdigest()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))

    def make_arrivals(scenario, count):
        offsets = arrival_offsets(count, scenario, sparse_interval=args.sparse_interval,
            saturated_interval=args.saturated_interval, burst_size=args.burst_size,
            burst_interval=args.burst_interval)
        workload_cases = [cases[index] for index in case_indices(count, scenario)]
        return [{'offset_s': offset, 'case': case['name'], 'series': len(case['data']['inputs']),
                 'encoded': json.dumps({'model': spec.model_id, 'task': 'plugin', 'data': case['data']},
                                       separators=(',', ':')).encode()}
                for offset, case in zip(offsets, workload_cases, strict=True)]

    @contextmanager
    def server(mode, directory, diagnostic):
        directory.mkdir()
        monitor.wait_idle()
        ensure_port_free('127.0.0.1', args.port)
        trace = directory / 'steps.jsonl'
        command, environment = build_vllm_command(
            adapter='timesfm25', port=args.port, benchmark=True,
            max_num_seqs=1 if mode == 'single' else 4,
            forecast_batch_rows=None if mode == 'single' else 8,
            continuous_batching=mode == 'continuous', forecast_queue_limit=args.queue_limit)
        if '--no-async-scheduling' not in command:
            command.append('--no-async-scheduling')
        command += ['--gpu-memory-utilization', '.35']
        environment.pop('VTSFM_CONTINUOUS_TRACE', None)
        if diagnostic:
            environment['VTSFM_CONTINUOUS_TRACE'] = str(trace)
            if mode != 'continuous':
                command += ['--scheduler-cls', 'vtsfm.continuous_scheduler.ForecastTracingScheduler']
        environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE='0',
                           OMP_NUM_THREADS='4', PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT)]))
        metadata['commands'][f'{mode}-{"diagnostic" if diagnostic else "primary"}'] = command
        with OwnedProcesses(directory) as owner, requests.Session() as control:
            control.trust_env = False
            child = owner.spawn(command, environment, 'server')
            wait_ready(child, url, control, timeout=180)
            # Both passes warm the same cases; numerical checks are outside replay.
            for index, case in enumerate(cases):
                response = control.post(url + '/pooling', json={
                    'model': spec.model_id, 'task': 'plugin', 'data': case['data']}, timeout=args.warmup_timeout)
                response.raise_for_status()
                predictions = response.json()['data']['predictions']
                expected = args.output / f'expected-{index}.json'
                if not expected.exists():
                    expected.write_text(json.dumps(predictions))
                else:
                    import numpy as np
                    baseline = json.loads(expected.read_text())
                    for actual, reference in zip(predictions, baseline, strict=True):
                        assert actual['item_id'] == reference['item_id']
                        for key, values in reference.items():
                            if key != 'item_id':
                                np.testing.assert_allclose(actual[key], values, rtol=1e-4, atol=1e-4)
            yield owner, child, trace
        monitor.wait_idle()

    try:
        monitor.assert_idle()
        url = f'http://127.0.0.1:{args.port}'
        for mode in modes:
            directory = args.output / mode
            directory.mkdir()
            primary_results = {}
            with server(mode, directory / 'primary', False) as (owner, child, trace):
                for scenario in scenarios:
                    if args.warmup_requests:
                        # Exercise shared cohorts before capturing primary counters/peaks.
                        warmup_path = directory / f'warmup-{scenario}.json'
                        warmup = replay(url, make_arrivals(scenario, args.warmup_requests),
                            model=spec.model_id, request_prefix=f'warmup-{mode}-{scenario}',
                            monitor=monitor, cancel=lambda: owner.stop(child),
                            record_path=warmup_path, timeout=args.warmup_timeout)
                        warmup.update(mode=mode, scenario=scenario, manifest_sha256=manifest_hash,
                                      instrumentation='untimed offered-arrival warmup; excluded from primary results')
                        warmup_path.write_text(json.dumps(warmup, indent=2) + '\n')
                        if warmup['completed'] != args.warmup_requests or warmup['failed'] or warmup['rejected']:
                            raise RuntimeError(f'{mode}/{scenario} warmup failed; keep raw evidence')
                        _drain_requests(url, directory / f'warmup-drain-{scenario}.json')
                    before = _queue_counter(url)
                    allocator_before = _allocator_stats(url, reset=True)
                    result = replay(url, make_arrivals(scenario, args.requests), model=spec.model_id,
                        request_prefix=f'{mode}-{scenario}', monitor=monitor,
                        cancel=lambda: owner.stop(child), record_path=directory / f'{scenario}.json')
                    after = _queue_counter(url)
                    result.update(mode=mode, scenario=scenario, manifest_sha256=manifest_hash,
                                  instrumentation='primary: no scheduler/worker trace writes',
                                  allocator_before=allocator_before,
                                  allocator_after=_allocator_stats(url),
                                  memory_method='warmed Torch allocated/reserved peaks plus 20 ms NVML; no empty_cache between points')
                    count = after.get('count', 0) - before.get('count', 0)
                    result['metrics_queue_count'] = count
                    result['metrics_queue_mean_ms'] = (
                        1000 * (after['sum'] - before['sum']) / count
                        if count > 0 and 'sum' in after and 'sum' in before else None)
                    (directory / f'{scenario}.json').write_text(json.dumps(result, indent=2) + '\n')
                    primary_results[scenario] = result
                    summary = {key: value for key, value in result.items() if key not in ('samples', 'nvml_samples')}
                    metadata['results'].append(summary)
                    (args.output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
                    print(json.dumps({key: value for key, value in result.items()
                                      if key not in ('samples', 'nvml_samples', 'queue_delay')}), flush=True)
                    if result['failed']:
                        raise RuntimeError(f'{mode}/{scenario} had failed requests; keep partial evidence')
            count = metadata['diagnostic_prefix_requests']
            if count:
                with server(mode, directory / 'diagnostic', True) as (owner, child, trace):
                    for scenario in scenarios:
                        result = replay(url, make_arrivals(scenario, count), model=spec.model_id,
                            request_prefix=f'diag-{mode}-{scenario}', monitor=monitor,
                            cancel=lambda: owner.stop(child), record_path=directory / 'diagnostic' / f'{scenario}.json')
                        result.update(instrumentation='separate traced diagnostic prefix',
                                      queue_delay=queue_delays(trace, result['samples']))
                        completed_ids = {row['request_id'] for row in result['samples'] if row['status'] == 'passed'}
                        measured_ids = {row['request_id'] for row in result['queue_delay']['samples']}
                        result['queue_delay']['missing_completed_request_ids'] = sorted(completed_ids - measured_ids)
                        (directory / 'diagnostic' / f'{scenario}.json').write_text(json.dumps(result, indent=2) + '\n')
                        diagnostic = {'offered_prefix': count, 'source': f'{mode}/diagnostic/{scenario}.json',
                                      'queue_delay': result['queue_delay']}
                        primary_results[scenario]['queue_delay_diag'] = diagnostic
                        (directory / f'{scenario}.json').write_text(json.dumps(primary_results[scenario], indent=2) + '\n')
                        next(row for row in metadata['results'] if row['mode'] == mode
                             and row['scenario'] == scenario)['queue_delay_diag'] = diagnostic
                        (args.output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
                        if result['failed']:
                            raise RuntimeError(f'{mode}/{scenario} diagnostic failed; keep partial evidence')
                        if completed_ids - measured_ids:
                            raise RuntimeError(f'{mode}/{scenario} lacks correlated queue diagnostics; see raw trace')
    finally:
        signal.signal(signal.SIGTERM, previous)
        monitor.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
