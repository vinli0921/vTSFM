"""Manual full-model retained-shape compilation parity gate; no timings."""

import importlib
import json
import math
import os
from pathlib import Path

import pytest


ADAPTER = os.environ.get('VTSFM_TEST_ADAPTER')
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_TEST_EXECUTION') != 'compile'
    or ADAPTER not in ('timesfm25', 'toto2', 'moirai2'),
    reason='Select --forecast-execution compile in scripts/test-models.sh on an idle GPU')]


def test_compiled_neural_core_matches_pristine_reference_after_input_and_shape_changes(monkeypatch):
    from benchmarks.run import GpuMonitor

    monitor = GpuMonitor(int(os.environ.get('VTSFM_TEST_GPU_INDEX', '0')))
    try:
        monitor.assert_idle()
    finally:
        monitor.close()
    # This extra regression is always FP32, even when the HTTP profile selects
    # BF16/TF32. Set both native/candidate modes before Torch initializes CUDA.
    monkeypatch.setenv('VTSFM_PRECISION', 'fp32')
    monkeypatch.setenv('VTSFM_TEST_PRECISION', 'fp32')
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    import torch
    from torch._dynamo import config
    from torch._dynamo.utils import counters

    from model_integration_support import NativeForecaster

    assert os.environ.get('TRITON_DEFAULT_FP_FUSION') == '0'
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')
    previous_math = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    previous_config = {name: getattr(config, name) for name in
                       ('recompile_limit', 'accumulated_recompile_limit',
                        'fail_on_recompile_limit_hit', 'suppress_errors')}
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        native = NativeForecaster(ADAPTER)
        cases = []
        for rows, context, horizon, offset in [(1, 512, 96, 0), (16, 2048, 800, 0),
                                               (16, 2048, 800, 5)]:
            histories = [[offset + row / 7 + math.sin(step / 17 + row)
                          for step in range(context)] for row in range(rows)]
            data = {'inputs': [{'target': history} for history in histories],
                    'parameters': {'prediction_length': horizon}}
            cases.append((histories, horizon, native.predict_array(data).copy()))
        configuration = json.loads((Path(native.metadata()['model_path']) / 'config.json').read_text())
        name = {'timesfm25': 'TimesFMRuntime', 'toto2': 'TotoRuntime', 'moirai2': 'MoiraiRuntime'}[ADAPTER]
        module = importlib.import_module(f'vtsfm.adapters.{ADAPTER}.runtime')
        # Exercise the actual constructor hook, then strict checkpoint loading.
        monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'compile')
        runtime = getattr(module, name)(configuration)
        reference_model = native.reference.model if ADAPTER == 'timesfm25' else native.reference
        runtime.model.load_state_dict(reference_model.state_dict(), strict=True)
        runtime.model.to(device=native.device, dtype=torch.float32).eval()
        del native, reference_model
        before = counters['stats']['unique_graphs']
        with torch.inference_mode():
            for index in (0, 1, 2, 1, 0):
                histories, horizon, expected = cases[index]
                context = torch.tensor(histories, device='cuda', dtype=torch.float32)
                original = context.clone()
                actual = runtime.forecast(context, horizon)
                torch.testing.assert_close(actual.cpu(), torch.as_tensor(expected), rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(context, original, rtol=0, atol=0)
        captured = counters['stats']['unique_graphs'] - before
        assert captured > 0, 'Numerical agreement alone does not establish compiler activity'
        print({'adapter': ADAPTER, 'forecast_execution': 'compile', 'captured_graphs_delta': captured})
    finally:
        torch.set_float32_matmul_precision(previous_math[0])
        torch.backends.cuda.matmul.allow_tf32 = previous_math[1]
        for name, value in previous_config.items():
            setattr(config, name, value)
