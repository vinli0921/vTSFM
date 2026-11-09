"""Manual current-stack parity gate for guarded neural compilation, not timing."""

import importlib
import json
import math
import os
from pathlib import Path

import pytest


ADAPTER = os.environ.get('VTSFM_TEST_ADAPTER')
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('VTSFM_RUN_COMPILED_FORECAST') != '1'
    or ADAPTER not in ('timesfm25', 'toto2', 'moirai2'),
    reason='Select VTSFM_TEST_ADAPTER and VTSFM_RUN_COMPILED_FORECAST=1 on the pinned idle GPU')]


def test_compiled_neural_core_matches_pristine_reference_after_input_and_shape_changes(monkeypatch):
    from benchmarks.run import GpuMonitor

    GpuMonitor(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))).assert_idle()
    import torch

    from model_integration_support import NativeForecaster
    from vtsfm.compilation import EXECUTION_ENV, configure_compilation

    # The shell must set contraction policy before this process imports Torch.
    assert os.environ.get('TRITON_DEFAULT_FP_FUSION') == '0'
    assert os.environ.get('NVIDIA_TF32_OVERRIDE') == '0'
    monkeypatch.setenv('VTSFM_PRECISION', 'fp32')
    monkeypatch.setenv(EXECUTION_ENV, 'eager')
    monkeypatch.delenv('VTSFM_CONTINUOUS_BATCHING', raising=False)
    previous = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    state = None
    try:
        native = NativeForecaster(ADAPTER)
        cases = []
        for rows, context, horizon, offset in [(1, 512, 96, 0), (16, 2048, 800, 0),
                                               (16, 2048, 800, 5)]:
            histories = [[offset + row / 7 + math.sin(step / 17 + row)
                          for step in range(context)] for row in range(rows)]
            data = {'inputs': [{'target': history} for history in histories],
                    'parameters': {'prediction_length': horizon}}
            # Capture every genuine reference before mutating the candidate.
            cases.append((histories, horizon, native.predict_array(data).copy()))
        config = json.loads((Path(native.metadata()['model_path']) / 'config.json').read_text())
        name = {'timesfm25': 'TimesFMRuntime', 'toto2': 'TotoRuntime', 'moirai2': 'MoiraiRuntime'}[ADAPTER]
        runtime = getattr(importlib.import_module(f'vtsfm.adapters.{ADAPTER}.runtime'), name)(config)
        reference_model = native.reference.model if ADAPTER == 'timesfm25' else native.reference
        runtime.model.load_state_dict(reference_model.state_dict())
        runtime.model.to(device=native.device, dtype=torch.float32).eval()
        del native, reference_model
        monkeypatch.setenv(EXECUTION_ENV, 'compile')
        before = int(torch._dynamo.utils.counters['stats']['unique_graphs'])
        state = configure_compilation(runtime, ADAPTER)
        with torch.inference_mode():
            for index in (0, 1, 2, 1, 0):
                histories, horizon, expected = cases[index]
                context = torch.tensor(histories, device='cuda', dtype=torch.float32)
                original = context.clone()
                actual = runtime.forecast(context, horizon)
                torch.testing.assert_close(actual.cpu(), torch.as_tensor(expected), rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(context, original, rtol=0, atol=0)
        captured = int(torch._dynamo.utils.counters['stats']['unique_graphs']) - before
        assert captured > 0, 'Numerical agreement alone does not establish compiler activity'
        assert state.calls > 0
        print({**state.describe(), 'captured_graphs_delta': captured})
    finally:
        if state is not None:
            state.close()
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
