"""Manual full-forecast CUDA-graph comparison for TimesFM, not HTTP serving.

Four fresh-process arms share the FixedForecast input/output wrapper:
dense-eager, paged-eager, dense-graph and paged-graph. Capture/setup is excluded
and reported separately; timed calls include validation, static input copy,
complete forecast execution/replay, owned output clone and CUDA synchronization.
Weights and any explicitly supplied AssetOpsBench fixture must already be cached.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

from benchmarks.timesfm_native_kv import (
    GPU_OWNERSHIP_METHOD, TOLERANCE, _assert_gpu_owned, _context, _data,
    _integers, _installed_metadata, _sha, _source_metadata, _trace_summary, _write,
)


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('dense-eager', 'paged-eager', 'dense-graph', 'paged-graph')
SUMMARY_KEYS = ('n', 'mean_ms', 'p50_ms', 'p95_ms', 'request_per_s', 'series_per_s')


def _sources():
    metadata = _source_metadata()
    metadata['source_sha256']['benchmarks/timesfm_full_graph.py'] = _sha(Path(__file__))
    return metadata


def _compare(torch, actual, expected, rows, horizon):
    if tuple(actual.shape) != (rows, 9, horizon):
        raise AssertionError(f'Unexpected forecast shape {tuple(actual.shape)}')
    value = actual.cpu()
    torch.testing.assert_close(value, expected, **TOLERANCE)
    return dict(TOLERANCE, passed=True, values_compared=value.numel(),
                quantile_channels=9, max_abs_error=float((value - expected).abs().max()))


def _profile(torch, executor, context, destination):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as diagnostic:
        output = executor.run(context)
        torch.cuda.synchronize()
    del output
    diagnostic.export_chrome_trace(str(destination))
    result = _trace_summary(destination)
    events = json.loads(destination.read_text())['traceEvents']
    launches = Counter(event.get('name', '') for event in events
                       if event.get('ph') == 'X'
                       and ('cudaGraphLaunch' in event.get('name', '')
                            or 'cuGraphLaunch' in event.get('name', '')))
    result.update(cuda_graph_launch_events=sum(launches.values()),
                  cuda_graph_launch_event_names=dict(launches))
    return result


def _point(args, runtime, torch, monitor, data, rows, length, horizon, destination):
    from benchmarks.common import summarize
    from benchmarks.run import Sampler

    backend, mode = args.backend.split('-')
    capture = mode == 'graph'
    record = {
        'backend': args.backend, 'executor_backend': backend, 'capture': capture,
        'repetition': args.child_repetition, 'rows': rows, 'context': length, 'horizon': horizon,
        'forecast_blocks': (horizon + 127) // 128,
        'requests': args.requests, 'warmups': args.warmups, 'setup_warmups': args.setup_warmups,
        'samples_s': [], 'status': 'running',
        'timing_boundary': 'Sequential in-process FixedForecast.run(A) plus CUDA synchronization: finite-input validation, static input copy, full forecast body/replay and owned output clone. Setup/capture, input preparation, A/B/A parity, GC, ownership queries and destruction are excluded.',
        'memory_method': 'Setup peaks are Torch allocated/reserved counters only; setup NVML peaks are not measured. After parity/warmup, GC and empty_cache outside timing, reset Torch peaks with graph/private pools and paged fixed storage still retained. Whole-device NVML sampled during timed trials every 20 ms can miss transients.',
        'cache_lifetime': 'Paged leases and fixed plans intentionally remain held for the FixedForecast owner lifetime, across eager calls or graph replays. Held slot storage is not a leak; close must return every lease and drop the physical pool.',
        'parity_inputs': 'A is the prepared/synthetic timed context; auxiliary B = -A*1.7 + 0.3 is a deterministic finite transform used only for replay/output-ownership checks, not observed-label accuracy.',
    }
    executor = sampler = None
    try:
        _assert_gpu_owned(monitor, record)
        from vtsfm.adapters.timesfm25.full_graph import FixedForecast

        context, record['data_windows'] = _context(torch, data, rows, length, horizon)
        second = -context * 1.7 + .3
        record['input_sha256'] = hashlib.sha256(context.cpu().numpy().tobytes()).hexdigest()
        record['auxiliary_input_sha256'] = hashlib.sha256(second.cpu().numpy().tobytes()).hexdigest()
        # Compute both whole-runtime reference forecasts before constructing
        # the fixed owner, which takes exclusive access for the paged arm.
        expected_a = runtime.forecast(context, horizon).cpu()
        expected_b = runtime.forecast(second, horizon).cpu()
        if torch.allclose(expected_a, expected_b, **TOLERANCE):
            raise AssertionError('A/B reference forecasts are too similar to test replay freshness')
        torch.cuda.synchronize()
        _assert_gpu_owned(monitor, record)
        gc.collect()
        torch.cuda.empty_cache()
        record['pre_setup_allocated_bytes'] = torch.cuda.memory_allocated()
        record['pre_setup_reserved_bytes'] = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        executor = FixedForecast(runtime, context, horizon, backend=backend,
                                 capture=capture, warmups=args.setup_warmups)
        torch.cuda.synchronize()
        record['setup_wall_s'] = time.perf_counter() - started
        record['setup'] = executor.setup
        record['setup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['setup_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        record['setup_nvml_peak_measured'] = False
        record['fixed_pool_bytes'] = executor.stepper.arena.cache_bytes if backend == 'paged' else 0
        record['actual_cuda_graph'] = isinstance(executor.graph, torch.cuda.CUDAGraph)
        if record['actual_cuda_graph'] != capture:
            raise AssertionError('Requested graph/eager mode does not match the actual graph object')
        _assert_gpu_owned(monitor, record)
        first = executor.run(context)
        saved_first = first.clone()
        record['parity'] = {'A_before': _compare(torch, first, expected_a, rows, horizon)}
        other = executor.run(second)
        record['parity']['B'] = _compare(torch, other, expected_b, rows, horizon)
        # Check before the final A could conceal a shared-output alias.
        if not torch.equal(first, saved_first):
            raise AssertionError('B replay overwrote the retained owned A output')
        again = executor.run(context)
        record['parity']['A_after'] = _compare(torch, again, expected_a, rows, horizon)
        _compare(torch, other, expected_b, rows, horizon)
        if not torch.equal(first, saved_first):
            raise AssertionError('A replay changed a previously returned owned output')
        record['retained_owned_outputs_unchanged'] = True
        del first, saved_first, other, again, expected_a, expected_b, second
        for _ in range(args.warmups):
            output = executor.run(context)
            del output
        torch.cuda.synchronize()
        _assert_gpu_owned(monitor, record)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        record['baseline_allocated_bytes'] = torch.cuda.memory_allocated()
        record['baseline_reserved_bytes'] = torch.cuda.memory_reserved()
        record['replay_count_before_trials'] = executor.replay_count
        sampler = Sampler(monitor, interval=.02)
        with sampler:
            for _ in range(args.requests):
                _assert_gpu_owned(monitor, record)
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = executor.run(context)
                torch.cuda.synchronize()
                record['samples_s'].append(time.perf_counter() - started)
                del output
                _assert_gpu_owned(monitor, record)
        # Preserve untraced peaks before optional profiler allocations exist.
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        record['replay_count_after_trials'] = executor.replay_count
        record['timed_graph_replays'] = executor.replay_count - record['replay_count_before_trials']
        arena = executor.stepper.arena if backend == 'paged' else None
        record['held_leased_cache_bytes_after_trials'] = arena.leased_cache_bytes if arena is not None else 0
        record['held_fixed_pool_bytes_after_trials'] = arena.cache_bytes if arena is not None else 0
        record['held_sequence_leases_after_trials'] = len(arena.allocated) if arena is not None else 0
        if arena is not None and (record['held_sequence_leases_after_trials'] != 2 * rows
                                  or not record['held_leased_cache_bytes_after_trials']):
            raise AssertionError('Fixed paged owner must retain both sign leases for every series')
        if record['timed_graph_replays'] != (args.requests if capture else 0):
            raise AssertionError('Actual timed graph replay count does not match the arm')
        record.update(summarize(record['samples_s'], rows))
        record['throughput_scope'] = 'Sequential in-process FixedForecast forecasts/s and series/s; not HTTP or cross-request scheduling throughput.'
        if args.profile:
            _assert_gpu_owned(monitor, record)
            torch.cuda.synchronize()
            record['profile'] = _profile(torch, executor, context, destination.with_suffix('.trace.json'))
            _assert_gpu_owned(monitor, record)
            if capture and not record['profile']['cuda_graph_launch_events']:
                raise AssertionError('The requested graph diagnostic contains no CUDA graph launch event')
        record['status'] = 'ok'
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        record.update(status='cancelled' if interrupted else 'oom' if isinstance(error, torch.OutOfMemoryError) else 'error',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        for key in SUMMARY_KEYS:
            record.pop(key, None)
        if interrupted:
            raise
    finally:
        try:
            if sampler is not None:
                record['nvml_samples'] = sampler.samples
                record['sampled_peak_gpu_bytes'] = max((sample[1] for sample in sampler.samples), default=None)
            record.setdefault('peak_allocated_bytes', torch.cuda.max_memory_allocated())
            record.setdefault('peak_reserved_bytes', torch.cuda.max_memory_reserved())
            if executor is not None:
                # Keep the arena reference after close removes its owned buffers,
                # so the raw point proves that pinned slots were actually freed.
                arena = executor.stepper.arena if backend == 'paged' else None
                try:
                    executor.close()
                finally:
                    record['leased_cache_bytes_after_close'] = arena.leased_cache_bytes if arena is not None else 0
                    record['fixed_pool_bytes_after_close'] = arena.cache_bytes if arena is not None else 0
                if record['leased_cache_bytes_after_close'] or record['fixed_pool_bytes_after_close']:
                    raise AssertionError('FixedForecast close left a paged lease or physical pool')
                record['closed'] = True
        except BaseException as error:
            record.update(status='error', cleanup_error=f'{type(error).__name__}: {error}')
            for key in SUMMARY_KEYS:
                record.pop(key, None)
            raise
        finally:
            _write(destination, record)
    return record


def _child(args):
    from benchmarks.run import GpuMonitor, ensure_worker_gpu

    monitor = GpuMonitor(args.gpu)
    try:
        monitor.assert_idle()  # precedes Torch/model imports and CUDA initialization
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime
        from vtsfm.adapters.timesfm25.spec import MODEL_ID, MODEL_REVISION

        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(0)
        ensure_worker_gpu({'gpu': {'uuid': getattr(properties, 'uuid', None)}}, monitor.uuid, require_uuid=True)
        directory = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                                      allow_patterns=['config.json', 'model.safetensors'])
        runtime = TimesFMRuntime({})
        runtime.model.load_state_dict(load_file(str(Path(directory) / 'model.safetensors')))
        runtime.model.cuda().float().eval()
        data, provenance = _data(args)
        metadata = dict(_sources(), backend=args.backend, repetition=args.child_repetition,
                        python=sys.version, platform=platform.platform(), gpu=monitor.metadata(),
                        gpu_ownership_method=GPU_OWNERSHIP_METHOD, packages=_installed_metadata(),
                        model_id=MODEL_ID, model_revision=MODEL_REVISION, torch_cuda_runtime=torch.version.cuda,
                        precision={'dtype': 'float32', 'matmul_precision': torch.get_float32_matmul_precision(),
                                   'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                                   'TRITON_F32_DEFAULT': os.environ.get('TRITON_F32_DEFAULT'),
                                   'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE'),
                                   'num_threads': torch.get_num_threads()},
                        geometry={key: getattr(runtime.model, key) for key in
                                  ('p', 'o', 'os', 'm', 'x', 'h', 'md', 'hd', 'q', 'aridx')},
                        data_provenance=provenance, results=[])
        _write(args.output / 'metadata.json', metadata)
        for length in args.contexts:
            for horizon in args.horizons:
                for rows in args.rows:
                    destination = args.output / f'C{length}-H{horizon}-R{rows}.json'
                    record = _point(args, runtime, torch, monitor, data, rows, length, horizon, destination)
                    compact = {key: value for key, value in record.items()
                               if key not in ('samples_s', 'nvml_samples', 'traceback')}
                    metadata['results'].append(compact)
                    _write(args.output / 'metadata.json', metadata)
                    print(json.dumps(compact), flush=True)
                    if record['status'] != 'ok':
                        return 1
                    gc.collect()
                    torch.cuda.empty_cache()
        return 0
    finally:
        monitor.close()


def _orchestrate(args):
    from benchmarks.run import OwnedProcesses

    manifest = dict(_sources(), benchmark='timesfm-full-forecast-graphs',
                    scope='FixedForecast numerical wrapper in fresh processes; no HTTP/EngineCore integration.',
                    gpu_ownership_method=GPU_OWNERSHIP_METHOD,
                    configuration={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()}, status='running', runs=[])
    _write(args.output / 'manifest.json', manifest)
    try:
        with OwnedProcesses(args.output) as owner:
            for repetition in range(args.repetitions):
                offset = repetition % len(ARMS)
                order = ARMS[offset:] + ARMS[:offset]
                for arm in order if args.backend == 'all' else (args.backend,):
                    destination = args.output / f'repetition-{repetition + 1}-{arm}'
                    command = [sys.executable, str(Path(__file__).resolve()), '--backend', arm,
                               '--output', str(destination), '--child-repetition', str(repetition + 1),
                               '--contexts', ','.join(map(str, args.contexts)),
                               '--horizons', ','.join(map(str, args.horizons)),
                               '--rows', ','.join(map(str, args.rows)), '--requests', str(args.requests),
                               '--warmups', str(args.warmups), '--setup-warmups', str(args.setup_warmups),
                               '--gpu', str(args.gpu)]
                    if args.data_dir is not None:
                        command += ['--data-dir', str(args.data_dir)]
                    if args.profile:
                        command.append('--profile')
                    environment = os.environ.copy()
                    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                                       NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
                                       PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT)]))
                    entry = {'backend': arm, 'repetition': repetition + 1, 'command': command,
                             'output': str(destination), 'status': 'running', 'started_at': time.time()}
                    manifest['runs'].append(entry)
                    _write(args.output / 'manifest.json', manifest)
                    try:
                        child = owner.spawn(command, environment, destination.name)
                        child.wait()
                        owner.stop(child)
                        entry.update(returncode=child.returncode,
                                     status='ok' if child.returncode == 0 else 'failed')
                    except BaseException as error:
                        entry.update(status='cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'error',
                                     error=f'{type(error).__name__}: {error}')
                        raise
                    finally:
                        entry['finished_at'] = time.time()
                        _write(args.output / 'manifest.json', manifest)
                    print(json.dumps(entry), flush=True)
    except BaseException as error:
        manifest.update(status='cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'error',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if manifest['status'] == 'running':
            manifest['status'] = 'ok' if all(run['status'] == 'ok' for run in manifest['runs']) else 'failed'
        manifest['finished_at'] = time.time()
        _write(args.output / 'manifest.json', manifest)
    return int(any(run['status'] != 'ok' for run in manifest['runs']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('all',) + ARMS, default='all')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a run')
    parser.add_argument('--contexts', type=_integers, default=(512,))
    parser.add_argument('--horizons', type=_integers, default=(1024,))
    parser.add_argument('--rows', type=_integers, default=(2,))
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=5, help='Wrapper calls before timed trials')
    parser.add_argument('--setup-warmups', type=int, default=3, help='Warmups inside FixedForecast construction/capture')
    parser.add_argument('--repetitions', type=int, default=4, help='Four rounds balance all four arm order positions')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-dir', type=Path, help='Existing prepared AssetOpsBench fixture; no downloads')
    parser.add_argument('--profile', action='store_true', help='Trace one additional warmed call after timing and count CUDA graph launch events')
    parser.add_argument('--child-repetition', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (min(args.requests, args.repetitions, args.setup_warmups) < 1 or args.warmups < 0
            or args.gpu < 0 or max(args.contexts) > 8192 or max(args.horizons) > 1024
            or max(args.rows) > 16):
        parser.error('Require requests/repetitions/setup-warmups >=1, warmups/gpu >=0, contexts<=8192, horizons<=1024, rows<=16')
    if args.child_repetition is not None and (args.backend == 'all' or args.child_repetition < 1):
        parser.error('Invalid internal child configuration')
    if os.environ.get('VTSFM_RUN_TIMESFM_NATIVE_KV') != '1':
        parser.error('Set VTSFM_RUN_TIMESFM_NATIVE_KV=1 to opt in to this manual GPU experiment')
    if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
        parser.error('Set TRITON_F32_DEFAULT=ieee before launching the benchmark')
    if args.child_repetition is not None and (os.environ.get('NVIDIA_TF32_OVERRIDE') != '0'
                                             or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu)):
        parser.error('Child requires NVIDIA_TF32_OVERRIDE=0 and CUDA_VISIBLE_DEVICES matching the GPU index')
    args.output = args.output.expanduser().resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    try:
        return _child(args) if args.child_repetition is not None else _orchestrate(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
