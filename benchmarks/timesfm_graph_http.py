"""Manual end-to-end HTTP comparison of four fixed-shape TimesFM modes.

The primary workload is B2/C512/H1024, all nine quantiles, one outstanding
keep-alive request. Each arm/round gets a fresh loopback vLLM server. Startup,
first observed request, numerical checks, fallback, and warmups are separate
from steady HTTP samples. Independent official-native references run first in
an owned subprocess, which exits before any server starts. No downloads occur.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback

from benchmarks.timesfm_native_kv import _data, _installed_metadata, _sha, _source_metadata, _write


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('dense-eager', 'paged-eager', 'dense-graph', 'paged-graph')
LEVELS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
ROWS, CONTEXT, HORIZON = 2, 512, 1024
OWNERSHIP_METHOD = (
    'NVML compute/graphics snapshots at phase boundaries and in the 20 ms memory '
    'sampler. Only the owned server/reference process or its descendants/process '
    'group may occupy the GPU. Queries are outside HTTP request timers. A detected '
    'foreign PID fails the point and stops queued requests; no foreign process is '
    'signalled. Snapshots cannot exclude short-lived activity between checks.'
)


def _sources():
    result = _source_metadata()
    for name in ('benchmarks/timesfm_graph_http.py', 'benchmarks/model_performance.py',
                 'tests/model_integration_support.py'):
        result['source_sha256'][name] = _sha(ROOT / name)
    return result


def _owned_process(pid, root_pid):
    """OwnedProcesses starts a fresh session; also accept descendant sessions."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root_pid:
            return True
        seen.add(pid)
        try:
            if os.getpgid(pid) == root_pid:
                return True
            # Linux stat permits spaces and parentheses in the command name.
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


class _GuardedMonitor:
    """Keep process ownership separate from the timed HTTP callable."""

    def __init__(self, monitor, child, record):
        self.monitor, self.child, self.record = monitor, child, record
        self.lock = threading.Lock()
        self.failure = None
        record['gpu_ownership'] = {
            'method': OWNERSHIP_METHOD, 'root_pid': child.pid, 'checks': 0,
            'foreign_pids': [], 'observations': [],
        }

    def assert_owned(self):
        with self.lock:
            data = self.record['gpu_ownership']
            observed = {'time': time.time()}
            data['checks'] += 1
            data['observations'].append(observed)
            try:
                nvml, gpu = self.monitor.nvml, self.monitor.gpu
                compute = nvml.nvmlDeviceGetComputeRunningProcesses(gpu)
                observed['compute_pids'] = sorted({p.pid for p in compute})
                try:
                    graphics = nvml.nvmlDeviceGetGraphicsRunningProcesses(gpu)
                    observed['graphics_supported'] = True
                except Exception as error:
                    unsupported = getattr(nvml, 'NVMLError_NotSupported', ())
                    if not unsupported or not isinstance(error, unsupported):
                        raise
                    graphics = []
                    observed['graphics_supported'] = False
                observed['graphics_pids'] = sorted({p.pid for p in graphics})
                pids = set(observed['compute_pids']) | set(observed['graphics_pids'])
                foreign = sorted(pid for pid in pids if not _owned_process(pid, self.child.pid))
                observed['foreign_pids'] = foreign
                observed['owned_pids'] = sorted(pids - set(foreign))
                data['foreign_pids'] = sorted(set(data['foreign_pids']) | set(foreign))
                if foreign:
                    raise RuntimeError(f'Foreign GPU processes {foreign}; invalidate this HTTP point')
            except Exception as error:
                observed['error'] = f'{type(error).__name__}: {error}'
                self.failure = observed['error']
                raise

    def used(self):
        self.assert_owned()
        return self.monitor.used()


class _StartupSession:
    def __init__(self, session, guarded):
        self.session, self.guarded = session, guarded

    def get(self, *args, **kwargs):
        self.guarded.assert_owned()
        return self.session.get(*args, **kwargs)


def _cases(args):
    """Use the same sensor/start windows as the in-process graph study, on CPU."""
    import numpy as np

    data, provenance = _data(args)
    if data is None:
        t = np.arange(CONTEXT, dtype=np.float32)
        values = np.stack([np.sin(t / (7 + row)) + np.float32(.3) * np.cos(t / 31)
                           + row / 10 + t / 1000 - np.float32(.5) for row in range(ROWS)]).astype(np.float32)
        windows = None
    else:
        available = data.shape[1] - CONTEXT - HORIZON + 1
        if available < 1:
            raise ValueError('Cached fixture is shorter than context plus reserved horizon')
        windows = [{'channel': row % data.shape[0], 'start': (row * 131) % available}
                   for row in range(ROWS)]
        values = np.stack([data[w['channel'], w['start']:w['start'] + CONTEXT] for w in windows])
    changed = -values * np.float32(1.7) + np.float32(.3)

    def case(name, rows, horizon):
        return {'name': name, 'data': {
            'inputs': [{'item_id': f'{name}-row-{row}', 'target': target.tolist()}
                       for row, target in enumerate(rows)],
            'parameters': {'prediction_length': horizon, 'quantile_levels': LEVELS},
        }}

    cases = [case('A', values, HORIZON), case('B-transformed', changed, HORIZON),
             case('A-new-ids', values, HORIZON),
             case('shape-fallback', values[:1, -128:], 33),
             case('A-after-fallback', values, HORIZON)]
    return cases, dict(provenance, selected_windows=windows,
                       timed_input_array_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                       auxiliary_input_array_sha256=hashlib.sha256(changed.tobytes()).hexdigest(),
                       auxiliary_notes='B = float32(-A*1.7 + 0.3); numerical replay check only, not observed data.')


def _rpc(session, url, method):
    response = session.post(url + '/collective_rpc', json={
        'method': method, 'args': [], 'kwargs': {}, 'timeout': 60}, timeout=90)
    response.raise_for_status()
    results = response.json()['results']
    if len(results) != 1 or not isinstance(results[0], dict):
        raise RuntimeError('This benchmark requires one GPU worker RPC result')
    return results[0]


def _route_state(state, mode, *, ready=False):
    required = {'mode', 'graph_replays', 'calls', 'fallbacks', 'capture_attempts',
                'setup', 'fixed_pool_bytes', 'leased_cache_bytes'}
    if not required <= state.keys() or state['mode'] != mode or not isinstance(state['setup'], dict):
        raise AssertionError(f'Wrong/uninitialized TimesFM serving mode: {state}')
    graph = mode.endswith('-graph')
    if state['capture_attempts'] != int(graph):
        raise AssertionError('Graph construction must happen exactly once during startup')
    if ready and any(state[name] for name in ('calls', 'graph_replays', 'fallbacks')):
        raise AssertionError('Readiness must precede forecast route calls/replays')
    if mode.startswith('paged-'):
        if not 0 < state['leased_cache_bytes'] <= state['fixed_pool_bytes']:
            raise AssertionError('Fixed paged serving owner must retain its leased pool')
    elif state['fixed_pool_bytes'] or state['leased_cache_bytes']:
        raise AssertionError('Dense mode unexpectedly owns paged storage')


def _route_delta(before, after, mode, *, calls, fallbacks=0):
    _route_state(after, mode)
    expected = {'calls': calls, 'graph_replays': calls if mode.endswith('-graph') else 0,
                'fallbacks': fallbacks, 'capture_attempts': 0}
    delta = {key: after[key] - before[key] for key in expected}
    if delta != expected:
        raise AssertionError(f'Wrong serving route delta: {delta}, expected {expected}')
    return delta


def _reference(args, owner, monitor, cases, environment):
    from benchmarks.run import ensure_worker_gpu
    from vtsfm.adapters import get_adapter

    spec = get_adapter('timesfm25')
    destination = args.output / 'native-reference.json'
    cases_file = args.output / 'cases.json'
    _write(cases_file, {'cases': cases})
    # A structured argv executes the official fixture after explicit FP32 flags.
    # The reference's model and forecast API are independent of graph wrappers.
    bootstrap = ('import runpy,sys,torch; '
                 'torch.set_float32_matmul_precision("highest"); '
                 'torch.backends.cuda.matmul.allow_tf32=False; '
                 'torch.backends.cudnn.allow_tf32=False; '
                 'sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [sys.executable, '-c', bootstrap, str(ROOT / 'tests/model_integration_support.py'),
               '--adapter', 'timesfm25', '--cases', str(cases_file), '--output', str(destination)]
    monitor.assert_idle()
    child = owner.spawn(command, environment, 'native-reference')
    record = {'status': 'running', 'command': command}
    guarded = _GuardedMonitor(monitor, child, record)
    started = time.perf_counter()
    try:
        deadline = time.monotonic() + args.startup_timeout
        while child.poll() is None:
            guarded.assert_owned()
            if time.monotonic() >= deadline:
                raise TimeoutError('Official native reference exceeded its startup/execution timeout')
            try:
                child.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
        if child.returncode:
            raise RuntimeError(f'Official native reference failed ({child.returncode}); inspect native-reference.log')
        record['elapsed_wall_s'] = time.perf_counter() - started
        native = json.loads(destination.read_text())
        if (native['model_id'], native['model_revision']) != (spec.model_id, spec.model_revision):
            raise AssertionError('Native checkpoint does not match serving')
        if [entry['case'] for entry in native['cases']] != cases:
            raise AssertionError('Native/HTTP cases differ')
        ensure_worker_gpu({'gpu': {'uuid': native['runtime']['uuid']}}, monitor.uuid)
        record.update(status='ok', runtime=native['runtime'])
        return native
    except BaseException as error:
        record.update(status='cancelled' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'error',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        owner.stop(child)
        _write(args.output / 'native-reference-run.json', record)
        monitor.wait_idle()


def _point(args, owner, monitor, mode, repetition, native, environment):
    import requests
    from benchmarks.model_performance import measure_requests
    from benchmarks.run import ensure_port_free, ensure_worker_gpu, memory, wait_ready
    from vtsfm.adapters import get_adapter
    from vtsfm.cli import build_vllm_command

    sys.path.insert(0, str(ROOT / 'tests'))
    from model_integration_support import compare_response

    spec = get_adapter('timesfm25')
    directory = args.output / f'repetition-{repetition}-{mode}'
    directory.mkdir()
    record = {'status': 'running', 'mode': mode, 'repetition': repetition,
              'rows': ROWS, 'context': CONTEXT, 'horizon': HORIZON, 'quantiles': LEVELS,
              'requests': args.requests, 'warmups': args.warmups, 'clients': 1,
              'smoke_only': args.requests < 100, 'parity': [],
              'timing_boundary': 'Pre-encoded HTTP POST through response JSON decode and light identity/schema checks, one persistent session and one outstanding request. Startup/cold request/native parity/fallback/warmups/worker RPC are outside steady timing.',
              'memory_method': 'Worker Torch allocated/reserved/peaks differ from NVML whole-device residency sampled every20ms. Retained graph/private and paged pools remain included. Lower Torch allocated alone is not a VRAM saving.',
              'throughput_scope': 'Successful HTTP completions divided by steady elapsed wall time. Sequential keep-alive client, not offered-load or concurrent cross-request batching.'}
    server = None
    stopped = False
    request_count = 0
    request_ids = set()

    def stop_server():
        nonlocal stopped
        if server is not None and not stopped:
            owner.stop(server)
            stopped = True

    monitor.assert_idle()
    ensure_port_free('127.0.0.1', args.port)
    command, server_environment = build_vllm_command(
        adapter='timesfm25', port=args.port, benchmark=True,
        max_num_seqs=1, timesfm_forecast_mode=mode)
    if '--no-async-scheduling' not in command:
        command.append('--no-async-scheduling')
    command += ['--gpu-memory-utilization', str(args.gpu_memory_utilization)]
    server_environment.update({key: environment[key] for key in
        ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NVIDIA_TF32_OVERRIDE', 'OMP_NUM_THREADS',
         'PYTHONPATH', 'TRITON_F32_DEFAULT', 'HF_HUB_OFFLINE', 'PYTHONUNBUFFERED')})
    record['command'] = command
    url = f'http://127.0.0.1:{args.port}'
    try:
        with requests.Session() as control, requests.Session() as client:
            control.trust_env = client.trust_env = False
            started = time.perf_counter()
            server = owner.spawn(command, server_environment, directory.name)
            guarded = _GuardedMonitor(monitor, server, record)
            wait_ready(server, url, _StartupSession(control, guarded), timeout=args.startup_timeout)
            record['startup_ready_s'] = time.perf_counter() - started
            ready = _rpc(control, url, 'bench_timesfm_forecast')
            _route_state(ready, mode, ready=True)
            record['route_at_readiness'] = ready
            record['worker_config'] = memory(control, url, 'vllm', 'config')
            ensure_worker_gpu(record['worker_config'], monitor.uuid)
            record['startup_worker_memory'] = memory(control, url, 'vllm', 'stats')
            record['startup_sampled_device_bytes'] = guarded.used()

            def send(encoded):
                nonlocal request_count
                if guarded.failure:
                    raise RuntimeError(guarded.failure)
                identity = f'graph-http-r{repetition}-{mode}-{request_count}'
                request_count += 1
                response = client.post(url + '/pooling', data=encoded,
                    headers={'Content-Type': 'application/json', 'X-Request-Id': identity}, timeout=args.request_timeout)
                response.raise_for_status()
                body = response.json()
                if body.get('request_id') != f'pooling-{identity}' or body['request_id'] in request_ids:
                    raise AssertionError('HTTP response request identity was stale or mismatched')
                request_ids.add(body['request_id'])
                return body

            def checked(entry, index):
                case = entry['case']
                payload = {'model': spec.model_id, 'task': 'plugin', 'data': case['data']}
                encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
                _write(directory / f'{index}-{case["name"]}-request.json', payload)
                guarded.assert_owned()
                start = time.perf_counter()
                body = send(encoded)
                elapsed = time.perf_counter() - start
                guarded.assert_owned()
                _write(directory / f'{index}-{case["name"]}-response.json', body)
                result = compare_response(body, case, entry['expected'])
                record['parity'].append(dict(result, status='passed', observed_http_latency_s=elapsed))
                return elapsed, encoded

            # A→B→A changes values and IDs on the real HTTP path. Readiness
            # already includes capture: this first POST is a cold observed client
            # request, not an estimate of graph capture duration.
            before = ready
            cold, encoded = checked(native['cases'][0], 0)
            record['cold_observed_http_s'] = cold
            checked(native['cases'][1], 1)
            checked(native['cases'][2], 2)
            after_aba = _rpc(control, url, 'bench_timesfm_forecast')
            record['route_delta_ABA'] = _route_delta(before, after_aba, mode, calls=3)
            checked(native['cases'][3], 3)
            after_fallback = _rpc(control, url, 'bench_timesfm_forecast')
            record['route_delta_shape_fallback'] = _route_delta(after_aba, after_fallback, mode, calls=0, fallbacks=1)
            checked(native['cases'][4], 4)
            after_return = _rpc(control, url, 'bench_timesfm_forecast')
            record['route_delta_return'] = _route_delta(after_fallback, after_return, mode, calls=1)
            for _ in range(args.warmups):
                guarded.assert_owned()
                send(encoded)
                guarded.assert_owned()
            before_trials = _rpc(control, url, 'bench_timesfm_forecast')
            record['route_before_trials'] = before_trials
            record['route_delta_warmups'] = _route_delta(after_return, before_trials, mode, calls=args.warmups)
            memory(control, url, 'vllm', 'empty_cache')
            record['worker_memory_before_trials'] = memory(control, url, 'vllm', 'reset')
            guarded.assert_owned()
            _write(directory / 'point.json', record)
            result = measure_requests(url, encoded, ROWS, requests=args.requests, clients=1,
                warmups=0, monitor=guarded, request=send, cancel=stop_server,
                record_path=directory / 'requests.json')
            record['measurement'] = result
            if result['status'] != 'passed':
                raise RuntimeError('A steady HTTP request failed; point is invalid')
            after_trials = _rpc(control, url, 'bench_timesfm_forecast')
            record['route_after_trials'] = after_trials
            record['route_delta_trials'] = _route_delta(before_trials, after_trials, mode, calls=args.requests)
            record['worker_memory_after_trials'] = memory(control, url, 'vllm', 'stats')
            guarded.assert_owned()
            record['successful_http_request_ids'] = sorted(request_ids)
            record['steady_request_payload_sha256'] = hashlib.sha256(encoded).hexdigest()
            record['status'] = 'ok'
    except BaseException as error:
        record.update(status='cancelled' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'error',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        # measure_requests persists interrupted/failed partial samples itself.
        raw = directory / 'requests.json'
        if raw.exists():
            record['measurement'] = json.loads(raw.read_text())
            record['measurement']['status'] = 'failed'
            for key in ('mean_ms', 'p50_ms', 'p95_ms', 'request_per_s', 'series_per_s'):
                record['measurement'][key] = None
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            raise
    finally:
        try:
            stop_server()
            monitor.wait_idle()
            record['gpu_idle_after_cleanup'] = True
        except BaseException as error:
            record.update(status='error', cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            _write(directory / 'point.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('all',) + ARMS, default='all')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='Existing pinned AssetOpsBench fixture; no download')
    parser.add_argument('--synthetic', action='store_true', help='Clearly labelled synthetic harness smoke instead of cached data')
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=4, help='Four rounds balance the four server order positions')
    parser.add_argument('--port', type=int, default=18141)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--startup-timeout', type=float, default=480)
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.35)
    args = parser.parse_args()
    if (args.requests < 1 or args.warmups < 0 or args.repetitions < 1 or args.gpu < 0
            or not 1 <= args.port <= 65535 or not 0 < args.gpu_memory_utilization < 1
            or any(not math.isfinite(t) or t <= 0 for t in (args.startup_timeout, args.request_timeout))):
        parser.error('Invalid counts, GPU, port, utilization or timeout')
    if os.environ.get('VTSFM_RUN_TIMESFM_NATIVE_KV') != '1':
        parser.error('Set VTSFM_RUN_TIMESFM_NATIVE_KV=1 to opt in to this manual GPU experiment')
    if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
        parser.error('Set TRITON_F32_DEFAULT=ieee before launching the benchmark')
    if platform.system() != 'Linux':
        parser.error('GPU HTTP measurements target Linux; help and fixture construction are CPU-safe')
    args.output = args.output.expanduser().resolve()
    args.data_dir = None if args.synthetic else args.data_dir.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(_sources(), benchmark='timesfm-four-mode-graph-http', status='running',
                    configuration={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()},
                    workload={'rows': ROWS, 'context': CONTEXT, 'horizon': HORIZON,
                              'quantiles': LEVELS, 'clients': 1},
                    ownership_method=OWNERSHIP_METHOD, runs=[])
    _write(args.output / 'manifest.json', manifest)
    monitor = None
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    try:
        from benchmarks.run import GpuMonitor, OwnedProcesses, ensure_port_free
        from vtsfm.adapters import get_adapter

        cases, provenance = _cases(args)
        spec = get_adapter('timesfm25')
        manifest.update(model_id=spec.model_id, model_revision=spec.model_revision,
                        packages=_installed_metadata(), data_provenance=provenance)
        ensure_port_free('127.0.0.1', args.port)
        monitor = GpuMonitor(args.gpu)
        monitor.assert_idle()
        manifest['gpu'] = monitor.metadata()
        _write(args.output / 'manifest.json', manifest)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
                           TRITON_F32_DEFAULT='ieee', PYTHONUNBUFFERED='1',
                           PYTHONPATH=os.pathsep.join([str(ROOT / 'tests'), str(ROOT / 'src'), str(ROOT)]),
                           VLLM_PLUGINS='')
        with OwnedProcesses(args.output) as owner:
            native = _reference(args, owner, monitor, cases, environment)
            manifest['native_runtime'] = native['runtime']
            for repetition in range(args.repetitions):
                offset = repetition % len(ARMS)
                order = ARMS[offset:] + ARMS[:offset]
                for mode in order if args.backend == 'all' else (args.backend,):
                    entry = {'mode': mode, 'repetition': repetition + 1, 'status': 'running'}
                    manifest['runs'].append(entry)
                    _write(args.output / 'manifest.json', manifest)
                    point = _point(args, owner, monitor, mode, repetition + 1, native, environment)
                    entry.update(status=point['status'], output=f'repetition-{repetition + 1}-{mode}/point.json')
                    if point['status'] == 'ok':
                        entry.update(mean_ms=point['measurement']['mean_ms'], p95_ms=point['measurement']['p95_ms'],
                                     request_per_s=point['measurement']['request_per_s'])
                    _write(args.output / 'manifest.json', manifest)
                    print(json.dumps(entry), flush=True)
                    if point['status'] != 'ok':
                        raise RuntimeError(f'HTTP point failed: {mode}; inspect {entry["output"]}')
        manifest['status'] = 'ok'
        return 0
    except BaseException as error:
        manifest.update(status='cancelled' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'error',
                        error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        try:
            if monitor is not None:
                monitor.close()
        finally:
            manifest['finished_at'] = time.time()
            _write(args.output / 'manifest.json', manifest)
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
