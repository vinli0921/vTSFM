"""TimesFM compile-scope selection and isolation from other serving paths."""

import json

import pytest

from vtsfm.cli import build_vllm_command, main


SCOPE_ENV = 'VTSFM_TIMESFM_COMPILE_SCOPE'


@pytest.mark.parametrize('scope', ['wide', 'whole'])
def test_experimental_scope_uses_fp32_and_keeps_contraction_off(monkeypatch, capsys, scope):
    monkeypatch.setenv(SCOPE_ENV, scope)
    monkeypatch.delenv('VTSFM_PRECISION', raising=False)
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '1')
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '1')
    main(['serve', '--adapter', 'timesfm25', '--forecast-execution', 'compile', '--dry-run'])
    result = json.loads(capsys.readouterr().out)
    assert result['environment'][SCOPE_ENV] == scope
    assert result['environment']['VTSFM_PRECISION'] == 'fp32'
    assert result['environment']['NVIDIA_TF32_OVERRIDE'] == '0'
    assert result['environment']['TRITON_DEFAULT_FP_FUSION'] == '0'


@pytest.mark.parametrize('scope,kwargs,error', [
    ('mistyped', {'forecast_execution': 'compile'}, SCOPE_ENV),
    ('wide', {'forecast_execution': 'eager'}, 'compile'),
    ('whole', {'forecast_execution': 'compile', 'forecast_precision': 'tf32'}, 'FP32'),
    ('wide', {'forecast_execution': 'compile', 'continuous_batching': True}, 'whole-forecast'),
])
def test_rejects_unknown_scope_and_untested_combinations(monkeypatch, scope, kwargs, error):
    monkeypatch.setenv(SCOPE_ENV, scope)
    with pytest.raises(ValueError, match=error):
        build_vllm_command(adapter='timesfm25', **kwargs)


def test_default_compile_scope_still_supports_continuous_batching(monkeypatch):
    monkeypatch.delenv(SCOPE_ENV, raising=False)
    command, env = build_vllm_command(
        adapter='timesfm25', forecast_execution='compile', continuous_batching=True)
    assert SCOPE_ENV not in env
    assert env['VTSFM_CONTINUOUS_BATCHING'] == '1'
    assert '--scheduler-cls' in command


def test_non_timesfm_adapters_ignore_timesfm_scope(monkeypatch):
    monkeypatch.setenv(SCOPE_ENV, 'whole')
    _, env = build_vllm_command(adapter='moirai2')
    assert SCOPE_ENV not in env


@pytest.mark.parametrize('scope,mode', [('positioned-eager', 'eager'), ('positioned', 'compile')])
def test_positioned_comparison_uses_matching_fp32_whole_forecast_paths(monkeypatch, scope, mode):
    monkeypatch.delenv('VTSFM_PRECISION', raising=False)
    monkeypatch.setenv(SCOPE_ENV, scope)
    _, environment = build_vllm_command(adapter='timesfm25', forecast_execution=mode)
    assert environment['VTSFM_PRECISION'] == 'fp32'
    assert environment['NVIDIA_TF32_OVERRIDE'] == '0'
    with pytest.raises(ValueError, match='whole-forecast'):
        build_vllm_command(adapter='timesfm25', forecast_execution=mode, continuous_batching=True)
    with pytest.raises(ValueError, match='requires'):
        build_vllm_command(adapter='timesfm25',
                           forecast_execution='eager' if mode == 'compile' else 'compile')


def test_positioned_control_rejects_combining_with_fixed_cuda_graph(monkeypatch):
    monkeypatch.setenv(SCOPE_ENV, 'positioned-eager')
    with pytest.raises(ValueError, match='standard whole-forecast'):
        build_vllm_command(adapter='timesfm25', forecast_execution='eager',
                           timesfm_forecast_mode='dense-graph')


def test_positioned_forecast_matches_eager_with_ragged_histories_and_reuse():
    torch = pytest.importorskip('torch')
    pytest.importorskip('timesfm')
    from types import SimpleNamespace

    from test_timesfm_block_decode import tiny_model
    from vtsfm.adapters.timesfm25.positioned_compile import PositionedForecast
    from vtsfm.adapters.timesfm25.runtime import TimesFMRuntime

    model = tiny_model()
    eager = object.__new__(TimesFMRuntime)
    eager.model = model
    student = PositionedForecast(SimpleNamespace(model=model), compiled=False)
    a = torch.arange(97, dtype=torch.float32).sin().unsqueeze(0)
    b = torch.arange(2 * 163, dtype=torch.float32).reshape(2, 163).cos() + 1
    b[0, :66] = torch.nan
    workloads = [(a, 96), (b, 257), (b - 2, 257), (a, 96)]
    preserved = first = None
    with torch.inference_mode():
        for context, horizon in workloads:
            original = context.clone()
            observed = torch.isfinite(context)
            prediction = student.run(context, observed, horizon)
            if first is None:
                first, preserved = prediction, prediction.clone()
            torch.testing.assert_close(prediction, eager.forecast(context, horizon), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(context, original, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(first, preserved, rtol=0, atol=0)
