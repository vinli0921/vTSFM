"""Run one fresh Toto A100 HTTP comparison arm on prepared AssetOpsBench windows.

Start each arm in a separate process. The native oracle exits before the server
starts. Timings exclude startup, first forecast per shape, parity and warmups.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from benchmarks.model_evaluation import assetops_cases
from benchmarks.model_performance import measure_requests
from benchmarks.run import GpuMonitor, OwnedProcesses, Sampler, ensure_port_free, ensure_worker_gpu, wait_ready
from benchmarks.timesfm_graph_http import _GuardedMonitor, _StartupSession, _reference, _rpc
from vtsfm.adapters.toto2.spec import COMPILE_SCOPE_ENV, MODEL_ID
from vtsfm.cli import build_vllm_command

ROOT = Path(__file__).resolve().parents[1]


def shape(text):
    try:
        rows, width, horizon = map(int, text.split(','))
        if min(rows, width, horizon) > 0:
            return rows, width, horizon
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('Expected positive rows,context,horizon')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def serve(args, monitor, owner, native, common, record, cache):
    import requests
    sys.path.insert(0, str(ROOT / 'tests'))
    from model_integration_support import compare_response

    ensure_port_free('127.0.0.1', args.port)
    # build_vllm_command validates the selected scope before starting vLLM.
    os.environ[COMPILE_SCOPE_ENV] = 'whole' if args.arm == 'whole' else 'core'
    command, environment = build_vllm_command(adapter='toto2', port=args.port, benchmark=True,
        forecast_execution='eager' if args.arm == 'eager' else 'compile', forecast_precision='fp32',
        max_num_seqs=1)
    environment.update(common, TORCHINDUCTOR_CACHE_DIR=str(Path(cache) / 'inductor'),
                       TRITON_CACHE_DIR=str(Path(cache) / 'triton'))
    command += ['--gpu-memory-utilization', '0.35', '--no-async-scheduling']
    record.update(command=command, cold_cache='fresh Inductor and Triton directories for this arm')
    url = f'http://127.0.0.1:{args.port}'
    opened = time.perf_counter()
    child = owner.spawn(command, environment, 'vllm')
    guard = _GuardedMonitor(monitor, child, record)
    try:
        with requests.Session() as session:
            session.trust_env = False
            wait_ready(child, url, _StartupSession(session, guard), timeout=args.startup_timeout)
            record['start_to_ready_s'] = time.perf_counter() - opened
            config = _rpc(session, url, 'bench_memory_config')
            ensure_worker_gpu(config, monitor.uuid, require_uuid=True)
            if config['cuda_matmul_allow_tf32'] or config['float32_matmul_precision'] != 'highest':
                raise AssertionError(f'Unexpected FP32 worker configuration: {config}')
            scope = _rpc(session, url, 'bench_toto_compile_scope')
            targets = [] if args.arm == 'eager' else [args.arm]
            if scope['compile_targets'] != targets or not scope['fast_unit_scaling'] or len(scope['installed_shims']) != 2:
                raise AssertionError(f'Wrong active Toto path: {scope}')
            if Path(scope['runtime_file']).resolve() != ROOT / 'src/vtsfm/adapters/toto2/runtime.py':
                raise AssertionError(f'Worker imported a different checkout: {scope}')
            record.update(worker_config=config, actual_scope=scope,
                          ready_counters=_rpc(session, url, 'bench_memory_stats'))
            for entry in native['cases']:
                case = entry['case']
                payload = {'model': MODEL_ID, 'task': 'plugin', 'data': case['data']}
                encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
                point = {'case': case['name'], 'payload_sha256': hashlib.sha256(encoded).hexdigest()}
                record['points'].append(point)
                guard.assert_owned()
                with Sampler(guard) as first_memory:
                    first = time.perf_counter()
                    response = session.post(url + '/pooling', data=encoded,
                                             headers={'Content-Type': 'application/json'}, timeout=args.startup_timeout)
                    response.raise_for_status()
                    body = response.json()
                    point['first_request_s'] = time.perf_counter() - first
                point['first_request_peak_residency_bytes'] = max(value for _, value in first_memory.samples)
                point['after_first_counters'] = _rpc(session, url, 'bench_memory_stats')
                point['parity'] = compare_response(body, case, entry['expected'])
                guard.assert_owned()
                for clients in args.clients:
                    label = f'{case["name"]}-k{clients}'
                    before = _rpc(session, url, 'bench_memory_reset')
                    result = measure_requests(url, encoded, len(case['data']['inputs']), clients=clients,
                        requests=args.requests, warmups=args.warmups, monitor=guard,
                        record_path=args.output / f'{label}.json')
                    after = _rpc(session, url, 'bench_memory_stats')
                    point.setdefault('timings', []).append({'clients': clients, 'raw': f'{label}.json',
                        'request_per_s': result['request_per_s'], 'p50_ms': result['p50_ms'],
                        'p95_ms': result['p95_ms'], 'peak_resident_gpu_bytes': result['peak_resident_gpu_bytes'],
                        'before': before, 'after': after})
                    if result['status'] != 'passed' or guard.failure:
                        raise RuntimeError(f'Invalid timing point {label}: {guard.failure}')
                    if args.arm != 'eager' and after['dynamo']['unique_graphs'] <= record['ready_counters']['dynamo']['unique_graphs']:
                        raise AssertionError('No numerical compiler activity after readiness')
                    print(f'{args.arm} {label}: {result["request_per_s"]:.2f} req/s; p95={result["p95_ms"]:.2f} ms', flush=True)
                    save(args.output / 'run.json', record)
    finally:
        owner.stop(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=['eager', 'core', 'whole'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--shapes', nargs='+', type=shape, default=[(1, 2048, 96), (4, 512, 96)])
    parser.add_argument('--clients', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--port', type=int, default=18572)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--startup-timeout', type=int, default=1200)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'arm': args.arm, 'status': 'running', 'points': [], 'started_utc_epoch': time.time(),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'working_tree': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()}
    # The owned-process context cleans up both reference and server on SIGTERM.
    def terminate(signal_number, frame):
        raise SystemExit(128 + signal_number)
    signal.signal(signal.SIGTERM, terminate)
    monitor = GpuMonitor(args.gpu)
    try:
        monitor.assert_idle()
        record['gpu'] = monitor.metadata()
        cases = [assetops_cases(args.data_dir, count=1, context=width, horizon=horizon, batch_size=rows)[0]
                 for rows, width, horizon in args.shapes]
        common = dict(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
            HF_HUB_OFFLINE='1', NVIDIA_TF32_OVERRIDE='0', TRITON_DEFAULT_FP_FUSION='0',
            OMP_NUM_THREADS='4', VTSFM_TOTO_FAST_UNIT_SCALING='1', PYTHONUNBUFFERED='1',
            PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]))
        reference_environment = dict(os.environ, **common)
        reference_environment.update(VTSFM_FORECAST_EXECUTION='eager', VTSFM_PRECISION='fp32')
        reference_environment.pop(COMPILE_SCOPE_ENV, None)
        with OwnedProcesses(args.output) as owner, TemporaryDirectory(prefix='vtsfm-toto-compile-') as cache:
            native = _reference(args, owner, monitor, cases, reference_environment, adapter='toto2')
            serve(args, monitor, owner, native, common, record, cache)
        monitor.wait_idle()
        record['status'] = 'passed'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        monitor.close()
        record['finished_utc_epoch'] = time.time()
        save(args.output / 'run.json', record)


if __name__ == '__main__':
    main()
