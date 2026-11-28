"""CPU-only checks for the TimesFM split-flip serving selection."""

import pytest

from vtsfm.adapters.timesfm25.spec import COMPILE_SCOPE_ENV, FLIP_LAYOUT_ENV
from vtsfm.cli import build_vllm_command
from vtsfm.precision import PRECISION_ENV, TF32_OVERRIDE_ENV


def test_split_flip_layout_allows_whole_compile(monkeypatch):
    monkeypatch.setenv(FLIP_LAYOUT_ENV, 'split')
    monkeypatch.setenv(COMPILE_SCOPE_ENV, 'whole')
    _, environment = build_vllm_command(adapter='timesfm25', forecast_execution='compile')
    assert environment[FLIP_LAYOUT_ENV] == 'split'
    assert environment[PRECISION_ENV] == 'fp32'
    assert environment[TF32_OVERRIDE_ENV] == '0'


def test_split_flip_layout_allows_eager_tf32(monkeypatch):
    monkeypatch.setenv(FLIP_LAYOUT_ENV, 'split')
    _, environment = build_vllm_command(adapter='timesfm25', forecast_precision='tf32')
    assert environment[FLIP_LAYOUT_ENV] == 'split'
    assert environment[PRECISION_ENV] == 'tf32'
    assert environment[TF32_OVERRIDE_ENV] == '1'


@pytest.mark.parametrize('scope,kwargs', [
    ('core', {'forecast_precision': 'tf32', 'forecast_execution': 'compile'}),
    ('core', {'continuous_batching': True}),
    ('core', {'timesfm_forecast_mode': 'dense-graph'}),
    ('core', {'forecast_batch_rows': 4, 'max_num_seqs': 2}),
    ('positioned-eager', {}),
])
def test_split_flip_layout_rejects_unqualified_settings(monkeypatch, scope, kwargs):
    monkeypatch.setenv(FLIP_LAYOUT_ENV, 'split')
    monkeypatch.setenv(COMPILE_SCOPE_ENV, scope)
    with pytest.raises(ValueError, match='split flips require independent whole-forecast serving'):
        build_vllm_command(adapter='timesfm25', **kwargs)


def test_other_adapters_ignore_timesfm_flip_layout(monkeypatch):
    monkeypatch.setenv(FLIP_LAYOUT_ENV, 'unsupported')
    with pytest.raises(ValueError, match=FLIP_LAYOUT_ENV):
        build_vllm_command(adapter='timesfm25')
    _, environment = build_vllm_command(adapter='toto2')
    assert FLIP_LAYOUT_ENV not in environment
