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
