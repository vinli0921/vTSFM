"""Compare a direct TimesFM compile scope against independent native predictions.

This diagnostic reports first-use and warm device-call latencies, then checks
all nine quantiles at the established tolerance. It has no HTTP timing or
profiling and should run once per fresh process on an idle GPU.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from benchmarks.model_evaluation import QUANTILES, signal_cases
from benchmarks.run import GpuMonitor, OwnedProcesses
from benchmarks.timesfm_graph_http import _GuardedMonitor, _reference
from benchmarks.timesfm_native_kv import _source_metadata, _write
from vtsfm.adapters.timesfm25.spec import COMPILE_SCOPE_ENV, COMPILE_SCOPES


ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('eager', *COMPILE_SCOPES)
TOLERANCE = 1e-4


def retained_cases():
    """Return short, recursive, changed recursive and ragged test histories."""
    cases = []
    designs = [
        ('short', [512], 96, 0),
        ('recursive', [2048] * 16, 800, 0),
        ('recursive-changed', [2048] * 16, 800, 5),
        ('ragged', [97, 163], 256, -2),
    ]
    for name, lengths, horizon, offset in designs:
        histories = [[offset + row / 7 + math.sin(step / 17 + row)
                      for step in range(length)] for row, length in enumerate(lengths)]
        cases.append({'name': f'synthetic-{name}', 'data': {
            'inputs': [{'target': row} for row in histories],
            'parameters': {'prediction_length': horizon, 'quantile_levels': list(QUANTILES)},
        }})
    return cases


def compare(actual, expected):
    import numpy as np

    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    result = {'expected_shape': list(expected.shape), 'actual_shape': list(actual.shape)}
    if actual.shape != expected.shape:
        return dict(result, status='wrong-shape')
    if not np.isfinite(expected).all():
        raise ValueError('Expected a finite independent reference')
    finite = np.isfinite(actual)
    errors = np.abs(actual - expected)
    limits = TOLERANCE + TOLERANCE * np.abs(expected)
    misses = ~finite | (errors > limits)
    return dict(result, status='failed' if misses.any() else 'passed',
                values=actual.size, misses=int(misses.sum()), nonfinite=int((~finite).sum()),
                max_abs_error=float(errors[finite].max()) if finite.any() else None,
                max_tolerance_multiple=float((errors[finite] / limits[finite]).max()) if finite.any() else None,
                misses_by_quantile=misses.sum(axis=(0, 2)).tolist(),
                misses_by_row=misses.sum(axis=(1, 2)).tolist(),
                misses_by_block=[int(misses[..., i:i + 128].sum()) for i in range(0, actual.shape[2], 128)])


def measure_case(runtime, entry, warm):
    import torch
    from torch._dynamo.utils import counters

    case = entry['case']
    data = case['data']
    histories = [item['target'] for item in data['inputs']]
    lengths = [len(history) for history in histories]
    context = torch.full((len(histories), max(lengths)), float('nan'), device='cuda')
    for padded, history in zip(context, histories, strict=True):
        padded[-len(history):] = torch.as_tensor(history, device=context.device, dtype=context.dtype)
    protected = context.clone()
    horizon = data['parameters']['prediction_length']
    compiled_before = counters['stats']['unique_graphs']
    torch.cuda.reset_peak_memory_stats()
    timings, parity = [], []
    with torch.inference_mode():
        for _ in range(1 + warm):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = runtime.forecast(context, horizon)
            torch.cuda.synchronize()
            timings.append(1000 * (time.perf_counter() - started))
            parity.append(compare(result.cpu().numpy(), entry['expected']))
            del result
    torch.testing.assert_close(context, protected, rtol=0, atol=0, equal_nan=True)
    return {
        'case': case['name'], 'input_lengths': lengths, 'left_nan_padding': [max(lengths)-n for n in lengths],
        'horizon': horizon, 'first_call_ms': timings[0], 'next_calls_ms': timings[1:],
        'new_unique_graphs': counters['stats']['unique_graphs'] - compiled_before, 'parity': parity,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    }


def worker(args):
    import torch
    from safetensors.torch import load_file
    from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime

    reference = json.loads((args.output / 'native-reference.json').read_text())
    model_path = Path(reference['model_path'])
    config = json.loads((model_path / 'config.json').read_text())
    runtime = TimesFMRuntime(config)
    runtime.model.load_state_dict(load_file(model_path / 'model.safetensors'), strict=True)
    runtime.model.to('cuda', dtype=torch.float32).eval()
    if runtime.compile_scope != args.scope:
        raise RuntimeError(f'Worker selected {runtime.compile_scope}, expected {args.scope}')
    result = {'scope': runtime.compile_scope, 'points': [], 'status': 'running'}
    entries = reference['cases']
    order = [0, 1, 2, 1, 3, 0, *range(4, len(entries))] if args.retained else list(range(len(entries)))
    for i in order:
        result['points'].append(measure_case(runtime, entries[i], args.warm))
        _write(args.output / 'candidate.json', result)
    numerics_ok = all(p['status'] == 'passed' for r in result['points'] for p in r['parity'])
    graphs = sum(r['new_unique_graphs'] for r in result['points'])
    result.update(status='passed' if numerics_ok else 'parity-failed', total_new_graphs=graphs)
    _write(args.output / 'candidate.json', result)
    if not numerics_ok:
        return 3
    if args.scope not in ('eager', 'positioned-eager') and graphs < 1:
        raise RuntimeError('No active compilation observed')
    return 0


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cases = retained_cases() if args.retained else []
    if args.data_dir is not None:
        cases.extend(signal_cases(args.data_dir))
    result = {'status': 'running', 'scope': args.scope, 'source': _source_metadata()}
    monitor = GpuMonitor(args.gpu)
    try:
        result['gpu'] = monitor.metadata()
        with OwnedProcesses(args.output) as owner, TemporaryDirectory(prefix='vtsfm-timesfm-compile-') as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       VTSFM_FORECAST_EXECUTION='eager', VTSFM_PRECISION='fp32',
                       VTSFM_TEST_PRECISION='fp32', NVIDIA_TF32_OVERRIDE='0',
                       TRITON_DEFAULT_FP_FUSION='0', HF_HUB_OFFLINE='1',
                       PYTHONPATH=os.pathsep.join(map(str, (ROOT / 'src', ROOT / 'tests', ROOT))),
                       TORCHINDUCTOR_CACHE_DIR=str(Path(cache) / 'inductor'),
                       TRITON_CACHE_DIR=str(Path(cache) / 'triton'))
            for key in (COMPILE_SCOPE_ENV, 'VTSFM_CONTINUOUS_BATCHING', 'VTSFM_TIMESFM_FORECAST_MODE'):
                env.pop(key, None)
            _reference(args, owner, monitor, cases, env)
            result['reference_sha256'] = hashlib.sha256((args.output / 'native-reference.json').read_bytes()).hexdigest()
            result['cases_sha256'] = hashlib.sha256((args.output / 'cases.json').read_bytes()).hexdigest()
            if args.scope != 'eager':
                env[COMPILE_SCOPE_ENV] = args.scope
                if args.scope != 'positioned-eager':
                    env['VTSFM_FORECAST_EXECUTION'] = 'compile'
            command = [sys.executable, '-m', 'benchmarks.timesfm_compile_gate', *sys.argv[1:], '--worker']
            monitor.assert_idle()
            child = owner.spawn(command, env, 'candidate')
            guard = _GuardedMonitor(monitor, child, result)
            result['gpu_ownership']['method'] = (
                'Controller snapshots of NVML compute and graphics owners about every 250 ms '
                'during the child lifetime. Device-call timers do not query NVML. Detection '
                'excludes the point and cleanup stops only owned subprocesses.'
            )
            deadline = time.monotonic() + args.startup_timeout
            while child.poll() is None:
                guard.assert_owned()
                if time.monotonic() >= deadline:
                    raise TimeoutError('TimesFM direct compile gate exceeded its execution timeout')
                try:
                    child.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            guard.assert_owned()
            state = {0: 'passed', 3: 'parity-failed'}.get(child.returncode)
            if state is None:
                raise RuntimeError(f'TimesFM worker exited {child.returncode}; inspect candidate.log')
            result['status'] = state
        monitor.wait_idle()
        return child.returncode
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        monitor.close()
        _write(args.output / 'run.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=SCOPES, required=True)
    parser.add_argument('--data-dir', type=Path, help='Prepared pinned AssetOpsBench signals directory')
    parser.add_argument('--retained', action='store_true', help='Also exercise retained/changed synthetic shapes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--startup-timeout', type=float, default=1200)
    parser.add_argument('--warm', type=int, default=3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if (args.data_dir is None and not args.retained) or args.warm < 1 or args.startup_timeout <= 0:
        parser.error('Select prepared signals and/or retained shapes, positive warm calls and positive timeout')
    if args.worker:
        return worker(args)

    def stop(number, _):
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, stop)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
