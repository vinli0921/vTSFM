"""Manual experimental TimesFM attention/KV comparison, in process rather than HTTP.

Run in the pinned TimesFM environment on an idle GPU, with
VTSFM_RUN_TIMESFM_NATIVE_KV=1 and TRITON_F32_DEFAULT=ieee set before launch.
Each backend/repetition gets a fresh process. Checkpoints and optional prepared
AssetOpsBench data must already be cached; this command downloads nothing.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('tensor', 'stepper', 'paged')
TOLERANCE = {'rtol': 1e-4, 'atol': 1e-4}


def _integers(value):
    try:
        result = tuple(int(part) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected comma-separated positive integers') from error
    if not result or any(n < 1 for n in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('Expected distinct positive integers')
    return result


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_metadata():
    paths = sorted((ROOT / 'src/vtsfm').rglob('*.py')) + [
        Path(__file__).resolve(), ROOT / 'benchmarks/run.py', ROOT / 'benchmarks/common.py']
    return {
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_dirty': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'source_sha256': {str(path.relative_to(ROOT)): _sha(path) for path in paths},
    }


def _installed_metadata():
    result = {}
    sources = {
        'timesfm': ('torch/transformer.py', 'torch/normalization.py', 'torch/util.py',
                    'timesfm_2p5/timesfm_2p5_torch.py', 'timesfm_2p5/timesfm_2p5_base.py'),
        'vllm': ('v1/attention/backends/triton_attn.py', 'v1/core/kv_cache_manager.py',
                 'v1/worker/utils.py', 'v1/kv_cache_interface.py',
                 'v1/attention/ops/triton_unified_attention.py'),
    }
    for package in ('torch', 'vllm', 'timesfm', 'triton', 'numpy', 'safetensors',
                    'huggingface-hub', 'nvidia-ml-py'):
        distribution = importlib.metadata.distribution(package)
        entry = {'version': distribution.version}
        direct = distribution.read_text('direct_url.json')
        if direct:
            entry['direct_url'] = json.loads(direct)
        entry['source_sha256'] = {
            name: _sha(Path(distribution.locate_file(f'{package}/{name}')))
            for name in sources.get(package, ())
        }
        result[package] = entry
    return result


def _data(args):
    """Read only an explicitly supplied, previously prepared fixture."""
    if args.data_dir is None:
        return None, {
            'kind': 'synthetic', 'seed': None,
            'formula': 'sin(t/(7+row)) + 0.3*cos(t/31) + row/10 + t/1000 - 0.5',
            'notes': 'Deterministic finite synthetic histories; repeated identical input per grid point. No accuracy claim.',
        }
    import numpy as np
    from benchmarks.prepare_data import COMMIT, SOURCE_SHA256

    manifest_path = args.data_dir / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = manifest.get('source', {})
    if (source.get('repository') != 'IBM/AssetOpsBench' or source.get('commit') != COMMIT
            or source.get('sha256') != SOURCE_SHA256):
        raise ValueError('Expected the pinned prepared AssetOpsBench manifest')
    path = args.data_dir / 'data.npz'
    if _sha(path) != manifest['array']['sha256']:
        raise ValueError('Prepared data checksum does not match its manifest')
    with np.load(path, allow_pickle=False) as archive:
        values = archive['series'].copy()
    if (values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all()
            or list(values.shape) != manifest['array']['shape']):
        raise ValueError('Expected finite float32 series matching the prepared manifest')
    return values, {
        'kind': 'observed_and_imputed', 'directory': str(args.data_dir),
        'manifest_sha256': _sha(manifest_path), 'data_sha256': _sha(path),
        'manifest': manifest,
        'notes': 'Correlated sensor/windows from one compressor; includes documented causal forward-fill. No accuracy claim.',
    }


def _context(torch, data, rows, length, horizon):
    if data is None:
        time_axis = torch.arange(length, dtype=torch.float32)
        value = torch.stack([torch.sin(time_axis / (7 + row)) + .3 * torch.cos(time_axis / 31)
                             + row / 10 + time_axis / 1000 - .5 for row in range(rows)])
        windows = None
    else:
        import numpy as np

        available = data.shape[1] - length - horizon + 1
        if available < 1:
            raise ValueError('Context plus reserved horizon exceeds the cached data length')
        windows = [{'channel': row % data.shape[0], 'start': (row * 131) % available}
                   for row in range(rows)]
        value = torch.from_numpy(np.stack([data[w['channel'], w['start']:w['start'] + length]
                                          for w in windows]))
    return value.to(device='cuda', dtype=torch.float32), windows


def _forecast(runtime, stepper, context, horizon):
    if stepper is None:
        return runtime.forecast(context, horizon)
    state = stepper.initialize(context, horizon)
    try:
        stepper.prefill([state])
        while not stepper.complete(state):
            stepper.advance([state])
        # Keep the returned tensor alive after release clears the state's copy.
        return stepper.finalize(state)
    finally:
        stepper.release(state)


def _trace_summary(path):
    events = json.loads(path.read_text())['traceEvents']
    kernels, copies = [], []
    for event in events:
        if event.get('ph') != 'X':
            continue
        categories = {category.strip().lower() for category in event.get('cat', '').split(',')}
        if 'kernel' in categories:
            kernels.append(event)
        if categories & {'gpu_memcpy', 'memcpy'}:
            copies.append(event)
    byte_counts = [event.get('args', {}).get('bytes') for event in copies]
    byte_counts = [value for value in byte_counts if isinstance(value, (int, float))]
    return {
        'trace_path': str(path), 'kernel_events': len(kernels),
        'kernel_duration_sum_us': sum(event.get('dur', 0) for event in kernels),
        'memcpy_events': len(copies), 'memcpy_events_with_bytes': len(byte_counts),
        'memcpy_bytes': sum(byte_counts) if byte_counts or not copies else None,
        'scope': 'One extra warmed forecast, traced diagnostic outside timed samples and NVML sampling. Kernel durations may overlap; their sum is not end-to-end latency. Memcpy counts use GPU copy events, excluding CPU API calls.',
    }


def _point(args, runtime, torch, monitor, data, rows, length, horizon, point_path):
    from benchmarks.common import summarize
    from benchmarks.run import Sampler

    capacity = (length + 31) // 32 + 4 * ((horizon - 1) // 128)
    record = {
        'backend': args.backend, 'repetition': args.child_repetition,
        'rows': rows, 'context': length, 'horizon': horizon,
        'forecast_blocks': (horizon + 127) // 128, 'cache_capacity_patches': capacity,
        'requests': args.requests, 'warmups': args.warmups,
        'samples_s': [], 'status': 'running',
        'timing_boundary': 'Serialized in-process forecast: initialize, all forecast blocks, finalize/release, and CUDA synchronization. Input preparation, parity, pool creation, GC, and pool destruction excluded.',
        'memory_method': 'After warmup, GC/empty_cache outside timing, then reset Torch peaks; fixed paged pool remains included. Whole-device NVML sampled every 20 ms may miss brief peaks.',
    }
    stepper = None
    sampler = None
    try:
        context, record['data_windows'] = _context(torch, data, rows, length, horizon)
        record['input_sha256'] = hashlib.sha256(context.cpu().numpy().tobytes()).hexdigest()
        # The owned model is still untouched here, including at later grid points.
        expected = runtime.forecast(context, horizon).cpu()
        if args.backend == 'paged':
            from vtsfm.adapters.timesfm25.native_stepping import PagedTimesFMStepper

            selected = PagedTimesFMStepper(runtime, max_sequences=rows, max_tokens=capacity)
        elif args.backend == 'stepper':
            from vtsfm.adapters.timesfm25.stepping import TimesFMStepper

            selected = TimesFMStepper(runtime)
        else:
            selected = None
        manager = selected if args.backend == 'paged' else nullcontext(selected)
        with manager as stepper:
            record['fixed_pool_bytes'] = stepper.arena.cache_bytes if args.backend == 'paged' else 0
            actual = _forecast(runtime, stepper, context, horizon).cpu()
            record['parity'] = dict(TOLERANCE, max_abs_error=float((actual - expected).abs().max()),
                                    values_compared=actual.numel(), quantile_channels=9, passed=False)
            if tuple(actual.shape) != (rows, 9, horizon):
                raise AssertionError(f'Unexpected forecast shape {tuple(actual.shape)}')
            torch.testing.assert_close(actual, expected, **TOLERANCE)
            record['parity']['passed'] = True
            del actual, expected
            for _ in range(args.warmups):
                output = _forecast(runtime, stepper, context, horizon)
                del output
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            record['baseline_allocated_bytes'] = torch.cuda.memory_allocated()
            record['baseline_reserved_bytes'] = torch.cuda.memory_reserved()
            sampler = Sampler(monitor, interval=.02)
            with sampler:
                for _ in range(args.requests):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = _forecast(runtime, stepper, context, horizon)
                    torch.cuda.synchronize()
                    record['samples_s'].append(time.perf_counter() - started)
                    del output
            # Capture measured peaks before optional profiler allocations exist.
            record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            record.update(summarize(record['samples_s'], rows))
            record['throughput_scope'] = 'In-process sequential requests/s and series/s; not HTTP or cross-request scheduling throughput.'
            record['leased_cache_bytes_after_trials'] = stepper.arena.leased_cache_bytes if args.backend == 'paged' else 0
            if record['leased_cache_bytes_after_trials']:
                raise AssertionError('Native KV leases remain after completed trials')
            if args.profile:
                from torch.profiler import ProfilerActivity, profile

                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as diagnostic:
                    output = _forecast(runtime, stepper, context, horizon)
                    torch.cuda.synchronize()
                del output
                trace_path = point_path.with_suffix('.trace.json')
                diagnostic.export_chrome_trace(str(trace_path))
                record['profile'] = _trace_summary(trace_path)
            record['status'] = 'ok'
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        status = 'cancelled' if interrupted else 'oom' if isinstance(error, torch.OutOfMemoryError) else 'error'
        record.update(status=status,
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if interrupted:
            raise
    finally:
        try:
            if sampler is not None:
                record['nvml_samples'] = sampler.samples
                record['sampled_peak_gpu_bytes'] = max((sample[1] for sample in sampler.samples), default=None)
            record.setdefault('peak_allocated_bytes', torch.cuda.max_memory_allocated())
            record.setdefault('peak_reserved_bytes', torch.cuda.max_memory_reserved())
            if stepper is not None and args.backend == 'paged':
                record['leased_cache_bytes_after_cleanup'] = stepper.arena.leased_cache_bytes
        except Exception as error:
            record['memory_collection_error'] = f'{type(error).__name__}: {error}'
            if record['status'] == 'ok':
                record['status'] = 'error'
        finally:
            # Persist even when SIGTERM/SystemExit interrupts an active trial.
            _write(point_path, record)
    return record


def _child(args):
    from benchmarks.run import GpuMonitor, ensure_worker_gpu

    monitor = GpuMonitor(args.gpu)
    try:
        # This must precede Torch imports and model/CUDA initialization.
        monitor.assert_idle()
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime
        from vtsfm.adapters.timesfm25.spec import MODEL_ID, MODEL_REVISION

        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('This manual benchmark requires CUDA')
        properties = torch.cuda.get_device_properties(0)
        ensure_worker_gpu({'gpu': {'uuid': str(getattr(properties, 'uuid', ''))}}, monitor.uuid)
        directory = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                                      allow_patterns=['config.json', 'model.safetensors'])
        runtime = TimesFMRuntime({})
        runtime.model.load_state_dict(load_file(str(Path(directory) / 'model.safetensors')))
        runtime.model.cuda().float().eval()
        data, provenance = _data(args)
        metadata = dict(_source_metadata(), backend=args.backend, repetition=args.child_repetition,
                        python=sys.version, platform=platform.platform(), gpu=monitor.metadata(),
                        packages=_installed_metadata(), model_id=MODEL_ID, model_revision=MODEL_REVISION,
                        torch_cuda_runtime=torch.version.cuda,
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
        failed = False
        for length in args.contexts:
            for horizon in args.horizons:
                for rows in args.rows:
                    point_path = args.output / f'C{length}-H{horizon}-R{rows}.json'
                    record = _point(args, runtime, torch, monitor, data, rows, length, horizon, point_path)
                    compact = {key: value for key, value in record.items()
                               if key not in ('samples_s', 'nvml_samples', 'traceback')}
                    metadata['results'].append(compact)
                    failed |= record['status'] != 'ok'
                    _write(args.output / 'metadata.json', metadata)
                    print(json.dumps(compact), flush=True)
                    # The point frame, stepper, and physical pool are gone first.
                    gc.collect()
                    torch.cuda.empty_cache()
        return int(failed)
    finally:
        monitor.close()


def _orchestrate(args):
    from benchmarks.run import OwnedProcesses

    manifest = dict(_source_metadata(), benchmark='experimental-timesfm-native-kv',
                    scope='Direct in-process tensor/stepper/backend comparison; no GPUModelRunner or HTTP integration.',
                    configuration={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()}, status='running', runs=[])
    _write(args.output / 'manifest.json', manifest)
    backends = BACKENDS + ('stepper-control',) if args.control_checkout else BACKENDS
    try:
        with OwnedProcesses(args.output) as owner:
            for repetition in range(args.repetitions):
                offset = repetition % len(backends)
                order = backends[offset:] + backends[:offset]
                for backend in order if args.backend == 'all' else (args.backend,):
                    destination = args.output / f'repetition-{repetition + 1}-{backend}'
                    source = args.control_checkout if backend == 'stepper-control' else ROOT
                    implementation = 'stepper' if backend == 'stepper-control' else backend
                    # An absolute script path and explicit PYTHONPATH select the
                    # control checkout even though OwnedProcesses retains our cwd.
                    command = [sys.executable, str(source / 'benchmarks/timesfm_native_kv.py'),
                               '--backend', implementation,
                               '--output', str(destination), '--child-repetition', str(repetition + 1),
                               '--contexts', ','.join(map(str, args.contexts)),
                               '--horizons', ','.join(map(str, args.horizons)),
                               '--rows', ','.join(map(str, args.rows)), '--requests', str(args.requests),
                               '--warmups', str(args.warmups), '--gpu', str(args.gpu)]
                    if args.data_dir is not None:
                        command += ['--data-dir', str(args.data_dir)]
                    if args.profile:
                        command.append('--profile')
                    environment = os.environ.copy()
                    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                                       NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4',
                                       PYTHONPATH=os.pathsep.join([str(source / 'src'), str(source)]))
                    entry = {'backend': backend, 'repetition': repetition + 1, 'command': command,
                             'source_checkout': str(source),
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
    return int(any(entry['status'] != 'ok' for entry in manifest['runs']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('all',) + BACKENDS, default='all')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a run')
    parser.add_argument('--contexts', type=_integers, default=(128, 512))
    parser.add_argument('--horizons', type=_integers, default=(384,))
    parser.add_argument('--rows', type=_integers, default=(1, 2), help='Independent series per sequential request')
    parser.add_argument('--requests', type=int, default=100, help='Timed sequential forecasts per grid point')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0, help='Physical NVML GPU index; children see only this GPU')
    parser.add_argument('--data-dir', type=Path, help='Existing prepared AssetOpsBench directory; no download')
    parser.add_argument('--control-checkout', type=Path,
                        help='Optional optimized-adapter checkout for a fourth, rotated stepper control; copy this exact harness there first')
    parser.add_argument('--profile', action='store_true', help='Save one additional warmed CPU/CUDA trace per point outside measurement')
    parser.add_argument('--child-repetition', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.requests < 1 or args.warmups < 0 or args.repetitions < 1 or args.gpu < 0
            or max(args.contexts) > 8192 or max(args.horizons) > 1024 or max(args.rows) > 16):
        parser.error('Require requests/repetitions >=1, warmups/gpu >=0, contexts <=8192, horizons <=1024, rows <=16')
    if args.child_repetition is not None and (args.backend == 'all' or args.child_repetition < 1):
        parser.error('Invalid internal child configuration')
    if os.environ.get('VTSFM_RUN_TIMESFM_NATIVE_KV') != '1':
        parser.error('Set VTSFM_RUN_TIMESFM_NATIVE_KV=1 to opt in to this manual GPU experiment')
    if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
        parser.error('Set TRITON_F32_DEFAULT=ieee before launching the benchmark')
    if args.control_checkout is not None:
        if args.backend != 'all' or args.child_repetition is not None:
            parser.error('--control-checkout requires the all-backend parent')
        args.control_checkout = args.control_checkout.expanduser().resolve()
        harness = args.control_checkout / 'benchmarks/timesfm_native_kv.py'
        if not harness.is_file() or _sha(harness) != _sha(Path(__file__)):
            parser.error('Control checkout must contain an identical benchmark harness')
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
