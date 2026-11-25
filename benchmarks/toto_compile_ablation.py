"""Isolate Toto's FP32 compilation and scaling arithmetic on the A100.

Each invocation loads the pinned independent native checkpoint and tests a single
clean candidate process. Timings are synchronized tensor calls, not HTTP.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCOPE = {
    'eager': ('eager', 'optimized'),
    'eager-native-scaling': ('eager', 'upstream'),
    'eager-tensor-scale': ('eager', 'device'),
    'eager-scale-multiply': ('eager', 'multiply'),
    'whole': ('whole', 'optimized'),
    'whole-tensor-scale': ('whole', 'device'),
    'whole-scale-multiply': ('whole', 'multiply'),
    'transformer': ('transformer', 'device'),
    'outer': ('outer', 'device'),
}
ROOT = Path(__file__).resolve().parents[1]


def summarize(actual, expected):
    """Use the unchanged native gate and count where a candidate exceeds it."""
    import numpy as np

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.ndim != 3:
        raise ValueError(f'Expected [row, quantile, horizon] arrays, got {actual.shape} and {expected.shape}')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Both forecasts must be finite')
    error = np.abs(actual - expected)
    ratio = error / (1e-4 + 1e-4 * np.abs(expected))
    bad = ratio > 1
    return {
        'checked': int(actual.size),
        'outside': int(bad.sum()),
        'outside_by_row': bad.sum(axis=(1, 2)).tolist(),
        'outside_by_quantile': bad.sum(axis=(0, 2)).tolist(),
        'max_abs': float(error.max()),
        'max_error_over_tolerance': float(ratio.max()),
        'max_native_magnitude_where_failed': float(np.abs(expected[bad]).max()) if bad.any() else None,
    }


def load_cases(path, query):
    document = json.loads(path.read_bytes())
    cases = document['cases'] if isinstance(document, dict) else document
    selected = [case for case in cases if query in case['name']]
    if not selected:
        raise ValueError(f'No case includes {query!r}')
    return selected


def build_runtime(native, arm):
    import torch

    from vtsfm.adapters.toto2 import fast_scaling
    from vtsfm.adapters.toto2.runtime import TotoRuntime
    from vtsfm.compilation import compile_method

    scope, scaling = SCOPE[arm]
    os.environ['VTSFM_FORECAST_EXECUTION'] = 'compile' if scope == 'whole' else 'eager'
    os.environ['VTSFM_TOTO_COMPILE_SCOPE'] = 'whole' if scope == 'whole' else 'core'
    os.environ['VTSFM_TOTO_FAST_UNIT_SCALING'] = '0' if scaling == 'upstream' else '1'
    if scaling in ('device', 'multiply'):
        def scale(input, fwd_scale, _bwd_scale):
            if fast_scaling._active.get() and not torch.is_grad_enabled():
                if scaling == 'device' and not isinstance(fwd_scale, torch.Tensor) and fwd_scale == 1.0:
                    return input
                return fwd_scale * input
            return None

        fast_scaling._fast = scale

    model_path = Path(native.metadata()['model_path'])
    configuration = json.loads((model_path / 'config.json').read_bytes())
    runtime = TotoRuntime(configuration)
    runtime.model.load_state_dict(native.reference.state_dict(), strict=True)
    runtime.model.cuda().eval()
    if scope == 'transformer':
        compile_method(runtime.model.transformer, 'forward')
    elif scope == 'outer':
        compile_method(runtime, '_forecast', ((runtime.model.transformer, 'forward'),))
    return runtime


def prepare_context(case):
    import torch
    from vtsfm.adapters.forecast import parse_request
    from vtsfm.adapters.toto2.spec import FORECAST_SETTINGS

    inputs, horizon, _ = parse_request(case['data'], FORECAST_SETTINGS)
    histories = [item['target'] for item in inputs]
    context = torch.full((len(histories), max(map(len, histories))), float('nan'),
                         dtype=torch.float32, device='cuda')
    for row, history in zip(context, histories, strict=True):
        row[-len(history):] = torch.tensor(history, dtype=torch.float32, device='cuda')
    return context, torch.isfinite(context), horizon


def execute(args, record):
    import numpy as np
    import torch
    from torch._dynamo.utils import counters

    sys.path.insert(0, str(ROOT / 'tests'))
    from model_integration_support import NativeForecaster, validate_cases

    cases = validate_cases('toto2', load_cases(args.cases, args.case_filter))
    native = NativeForecaster('toto2')
    references = [native.predict_array(case['data']).copy() for case in cases]
    record['native'] = native.metadata()
    if args.archived_native:
        archived = json.loads(args.archived_native.read_bytes())['cases']
        archive = {entry['case']['name']: entry for entry in archived}
        for case, expected in zip(cases, references, strict=True):
            entry = archive[case['name']]
            if entry['case']['data'] != case['data']:
                raise AssertionError(f'Archived input does not match {case["name"]}')
            np.testing.assert_array_equal(expected, np.asarray(entry['expected'], dtype=np.float32))
        record['archived_native_match'] = True
    runtime = build_runtime(native, args.arm)
    del native

    with torch.inference_mode():
        for case, expected in zip(cases, references, strict=True):
            context, observed, horizon = prepare_context(case)

            def forecast():
                output = runtime._forecast(context, observed, horizon)
                torch.cuda.synchronize()
                return output

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = forecast()
            first_call_s = time.perf_counter() - start
            first = summarize(output.cpu().numpy(), expected)
            for _ in range(3):
                forecast()
            samples_ms = []
            for _ in range(args.samples):
                start = time.perf_counter_ns()
                output = forecast()
                samples_ms.append((time.perf_counter_ns() - start) / 1e6)
            last = summarize(output.cpu().numpy(), expected)
            record['cases'].append({
                'name': case['name'],
                'shape': [*context.shape, horizon],
                'native_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                'first': first,
                'last': last,
                'first_call_s': first_call_s,
                'samples_ms': samples_ms,
                'p50_ms': float(np.percentile(samples_ms, 50)),
                'p95_ms': float(np.percentile(samples_ms, 95)),
                'dynamo_stats': dict(counters['stats']),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(SCOPE), required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--case-filter', default='condenser_water_flow')
    parser.add_argument('--archived-native', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    if os.environ.get('TRITON_DEFAULT_FP_FUSION') != '0' or os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        parser.error('Set TRITON_DEFAULT_FP_FUSION=0 and NVIDIA_TF32_OVERRIDE=0 before launching')
    os.environ['VTSFM_PRECISION'] = 'fp32'
    os.environ['VTSFM_TEST_PRECISION'] = 'fp32'
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    record = {
        'revision': revision, 'arm': args.arm, 'status': 'running', 'cases': [],
        'cases_file_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
    }
    try:
        execute(args, record)
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='error', error=repr(error))
        raise
    finally:
        with args.output.open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')


if __name__ == '__main__':
    main()
