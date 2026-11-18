"""Opt-in univariate Moirai attention (issue #24).

The first tests are CPU-safe and run everywhere. The rest build a tiny randomly
initialised ``Moirai2Module`` and need Torch, einops and uni2ts; they skip in the
lightweight CPU environment and run in the Moirai environment:

    PYTHONPATH=src:tests .venv-moirai2/bin/python -m pytest tests/test_moirai_attention.py -q

Random-weight parity can pass vacuously if attention has no effect on the output.
``test_negative_control_without_causal_mask_fails`` guards against that: dropping
the causal mask must change the forecast well beyond the parity tolerance.
"""

import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vtsfm.adapters.moirai2.spec import MOIRAI_ATTENTION_ENV, resolve_attention_mode
from vtsfm.cli import build_vllm_command

TOL = dict(rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# CPU-safe: mode selection
# ---------------------------------------------------------------------------

def test_attention_mode_defaults_to_upstream():
    assert resolve_attention_mode({}) == 'upstream'
    assert resolve_attention_mode({MOIRAI_ATTENTION_ENV: 'univariate'}) == 'univariate'


@pytest.mark.parametrize('value', ['', 'fused', 'UNIVARIATE', 'flex'])
def test_unknown_attention_mode_is_rejected(value):
    with pytest.raises(ValueError, match=MOIRAI_ATTENTION_ENV):
        resolve_attention_mode({MOIRAI_ATTENTION_ENV: value})


@pytest.mark.parametrize('execution, precision', [
    ('compile', None), ('compile', 'fp32'), ('compile', 'bf16'), ('compile', 'tf32'),
    ('eager', 'bf16'), ('eager', 'tf32'),
])
def test_cli_rejects_unqualified_univariate_attention(monkeypatch, execution, precision):
    monkeypatch.setenv(MOIRAI_ATTENTION_ENV, 'univariate')
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', execution)
    if precision is not None:
        monkeypatch.setenv('VTSFM_PRECISION', precision)
    else:
        monkeypatch.delenv('VTSFM_PRECISION', raising=False)
    with pytest.raises(ValueError, match='requires eager execution and FP32'):
        build_vllm_command(adapter='moirai2')


def test_cli_explicit_modes_take_precedence_for_univariate_attention(monkeypatch):
    monkeypatch.setenv(MOIRAI_ATTENTION_ENV, 'univariate')
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'compile')
    monkeypatch.setenv('VTSFM_PRECISION', 'bf16')
    _, environment = build_vllm_command(
        adapter='moirai2', forecast_execution='eager', forecast_precision='fp32')
    assert environment[MOIRAI_ATTENTION_ENV] == 'univariate'
    assert environment['VTSFM_FORECAST_EXECUTION'] == 'eager'
    assert environment['VTSFM_PRECISION'] == 'fp32'
    assert environment['NVIDIA_TF32_OVERRIDE'] == '0'


def test_cli_pins_fp32_for_univariate_attention_without_explicit_precision(monkeypatch):
    monkeypatch.setenv(MOIRAI_ATTENTION_ENV, 'univariate')
    monkeypatch.delenv('VTSFM_PRECISION', raising=False)
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '1')
    _, environment = build_vllm_command(adapter='moirai2', forecast_execution='eager')
    assert environment['VTSFM_PRECISION'] == 'fp32'
    assert environment['NVIDIA_TF32_OVERRIDE'] == '0'
    assert 'VTSFM_PRECISION' not in os.environ


@pytest.mark.parametrize('execution, precision, override, expected', [
    ('compile', '', '0', 'requires eager execution and FP32'),
    ('eager', 'bf16', '0', 'requires eager execution and FP32'),
    ('eager', 'tf32', '1', 'requires eager execution and FP32'),
    ('eager', '', '1', 'conflicts with fp32'),
])
def test_runtime_rejects_unsupported_attention_before_model_construction(
        execution, precision, override, expected):
    pytest.importorskip('torch')
    # A stand-in constructor makes the startup boundary observable without a
    # checkpoint or uni2ts installation. Run in a fresh process for the env config.
    script = """
import os
import sys
from types import ModuleType

execution, precision, override, expected = sys.argv[1:]
os.environ.update(VTSFM_MOIRAI_ATTENTION='univariate', VTSFM_FORECAST_EXECUTION=execution,
                  NVIDIA_TF32_OVERRIDE=override, TRITON_DEFAULT_FP_FUSION='0')
if precision:
    os.environ['VTSFM_PRECISION'] = precision
else:
    os.environ.pop('VTSFM_PRECISION', None)

upstream = ModuleType('uni2ts.model.moirai2')
def construct(**config):
    raise AssertionError('Loaded model before validating its execution settings')
upstream.Moirai2Module = construct
upstream.Moirai2Forecast = object
sys.modules[upstream.__name__] = upstream

from vtsfm.adapters.moirai2.runtime import MoiraiRuntime
try:
    MoiraiRuntime({})
except ValueError as error:
    assert expected in str(error), str(error)
else:
    raise AssertionError('Expected ValueError at runtime startup')
"""
    result = subprocess.run([sys.executable, '-c', script, execution, precision, override, expected],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Torch + uni2ts: tiny random model
# ---------------------------------------------------------------------------

TINY = dict(d_model=128, d_ff=256, num_layers=2, patch_size=16, max_seq_len=64,
            attn_dropout_p=0.0, dropout_p=0.0, num_predict_token=4)


@pytest.fixture(scope='module')
def lib():
    torch = pytest.importorskip('torch')
    pytest.importorskip('einops')
    moirai2 = pytest.importorskip('uni2ts.model.moirai2')
    attention = pytest.importorskip('vtsfm.adapters.moirai2.attention')
    return SimpleNamespace(torch=torch, moirai2=moirai2, attention=attention)


@pytest.fixture
def module(lib):
    lib.torch.manual_seed(0)
    return lib.moirai2.Moirai2Module(**TINY).eval()


def _context(lib, rows, length, pad=()):
    torch = lib.torch
    values = [[3.0 + row + math.sin(step / 5 + row) + 0.3 * math.cos(step / 11) + 0.01 * step
               for step in range(length)] for row in range(rows)]
    context = torch.tensor(values, dtype=torch.float32)
    for row, missing in pad:
        context[row, :missing] = float('nan')
    return context


def _forecast(lib, module, context, horizon):
    torch = lib.torch
    with torch.inference_mode():
        observed = torch.isfinite(context)
        runner = lib.moirai2.Moirai2Forecast(
            module=module, prediction_length=horizon, context_length=context.shape[-1], target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        return runner(past_target=context.masked_fill(~observed, 0).unsqueeze(-1),
                      past_observed_target=observed.unsqueeze(-1), past_is_pad=~observed)


CASES = {
    'single-pass H64': (3, 128, 64, ()),
    'recursive H96': (3, 128, 96, ()),
    'ragged left padding': (3, 128, 96, ((0, 40), (1, 17))),
    'patch boundary C17 H17': (2, 17, 17, ()),
}


@pytest.mark.parametrize('case', list(CASES))
def test_univariate_attention_matches_upstream(lib, module, case):
    rows, length, horizon, pad = CASES[case]
    context = _context(lib, rows, length, pad)
    expected = _forecast(lib, module, context, horizon)
    assert float(expected.std()) > 1e-3, 'degenerate reference forecast'
    uninstall = lib.attention.install_univariate_attention(module, verify=True)
    try:
        actual = _forecast(lib, module, context, horizon)
    finally:
        uninstall()
    lib.torch.testing.assert_close(actual, expected, **TOL)


def test_negative_control_without_causal_mask_fails(lib, module, monkeypatch):
    context = _context(lib, 3, 128, ((0, 40),))
    expected = _forecast(lib, module, context, 96)
    functional = lib.torch.nn.functional

    def unmasked(query, key, value, attn_mask=None, is_causal=False, **kwargs):
        return functional.scaled_dot_product_attention(query, key, value, **kwargs)

    monkeypatch.setattr(lib.attention, 'F', SimpleNamespace(scaled_dot_product_attention=unmasked))
    uninstall = lib.attention.install_univariate_attention(module)
    try:
        broken = _forecast(lib, module, context, 96)
    finally:
        uninstall()
    assert float((broken - expected).abs().max()) > 1e-2


def test_variate_bias_is_skipped_only_when_patched(lib, module, monkeypatch):
    from uni2ts.module.position import BinaryAttentionBias

    calls = []
    original = BinaryAttentionBias.forward
    monkeypatch.setattr(BinaryAttentionBias, 'forward',
                        lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    context = _context(lib, 2, 64)
    _forecast(lib, module, context, 64)
    assert calls, 'upstream path should compute the variate bias'
    calls.clear()
    uninstall = lib.attention.install_univariate_attention(module)
    try:
        _forecast(lib, module, context, 64)
    finally:
        uninstall()
    assert not calls


def test_unpadded_rows_use_is_causal_and_padded_rows_pass_a_bool_mask(lib, module, monkeypatch):
    functional = lib.torch.nn.functional
    seen = []

    def spy(query, key, value, attn_mask=None, is_causal=False, **kwargs):
        seen.append((query.dim(), is_causal, None if attn_mask is None else (attn_mask.dtype, attn_mask.shape[1])))
        return functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                                       is_causal=is_causal, **kwargs)

    monkeypatch.setattr(lib.attention, 'F', SimpleNamespace(scaled_dot_product_attention=spy))
    uninstall = lib.attention.install_univariate_attention(module)
    try:
        _forecast(lib, module, _context(lib, 2, 64), 64)
        assert seen and all(entry == (4, True, None) for entry in seen)
        seen.clear()
        _forecast(lib, module, _context(lib, 2, 64, ((0, 20),)), 64)
        assert seen and all(entry == (4, False, (lib.torch.bool, 1)) for entry in seen)
    finally:
        uninstall()


def test_multivariate_rows_raise_without_verify(lib, module):
    torch = lib.torch
    attention = module.encoder.layers[0].self_attn
    uninstall = lib.attention.install_univariate_attention(module)
    try:
        x = torch.randn(1, 4, TINY['d_model'])
        var_id = torch.tensor([[0, 0, 1, 1]])
        time_id = torch.tensor([[0, 1, 0, 1]])
        with torch.inference_mode(), pytest.raises(ValueError, match='exact only for univariate input'):
            attention(x, x, x, query_var_id=var_id, kv_var_id=var_id,
                      query_time_id=time_id, kv_time_id=time_id)
        single = torch.zeros(1, 4, dtype=torch.long)
        with torch.inference_mode():
            attention(x, x, x, query_var_id=single, kv_var_id=single, query_time_id=time_id, kv_time_id=time_id)
    finally:
        uninstall()


def test_variate_check_runs_once_per_forward_pass(lib, module, monkeypatch):
    calls = []
    original = lib.attention._VariateCheck.check

    def counting(self, *ids):
        cached = self.ref is not None and ids and self.ref() is ids[0]
        calls.append(not cached)
        return original(self, *ids)

    monkeypatch.setattr(lib.attention._VariateCheck, 'check', counting)
    uninstall = lib.attention.install_univariate_attention(module)
    try:
        _forecast(lib, module, _context(lib, 2, 64), 64)  # one pass, two layers
    finally:
        uninstall()
    assert len(calls) == TINY['num_layers'] and sum(calls) == 1


def test_reused_id_buffer_is_revalidated_on_every_module_call(lib, module):
    torch = lib.torch
    tokens, patch = 4, TINY['patch_size']
    target = torch.randn(1, tokens, patch)
    observed = torch.ones(1, tokens, patch, dtype=torch.bool)
    sample_id = torch.ones(1, tokens, dtype=torch.long)
    time_id = torch.arange(tokens).unsqueeze(0)
    variate_id = torch.zeros(1, tokens, dtype=torch.long)
    prediction_mask = torch.zeros(1, tokens, dtype=torch.bool)
    uninstall = lib.attention.install_univariate_attention(module)
    try:
        with torch.inference_mode():
            module(target, observed, sample_id, time_id, variate_id, prediction_mask, training_mode=False)
            variate_id[0, 2:] = 1  # same buffer, now multivariate
            with pytest.raises(ValueError, match='exact only for univariate input'):
                module(target, observed, sample_id, time_id, variate_id, prediction_mask, training_mode=False)
    finally:
        uninstall()
    assert not module._forward_pre_hooks, 'uninstall should remove the cache-reset hook'


def test_verify_rejects_more_than_one_variate(lib, module):
    torch = lib.torch
    attention = module.encoder.layers[0].self_attn
    uninstall = lib.attention.install_univariate_attention(module, verify=True)
    try:
        x = torch.randn(1, 4, TINY['d_model'])
        var_id = torch.tensor([[0, 0, 1, 1]])
        time_id = torch.tensor([[0, 1, 0, 1]])
        with torch.inference_mode(), pytest.raises(ValueError, match='variate'):
            attention(x, x, x, query_var_id=var_id, kv_var_id=var_id,
                      query_time_id=time_id, kv_time_id=time_id)
    finally:
        uninstall()


def test_source_guard_and_uninstall(lib, module, monkeypatch):
    assert lib.attention.upstream_forward_digest() == lib.attention.PINNED_FORWARD_SHA256
    context = _context(lib, 2, 64)
    expected = _forecast(lib, module, context, 64)
    uninstall = lib.attention.install_univariate_attention(module)
    with pytest.raises(RuntimeError, match='already'):
        lib.attention.install_univariate_attention(module)
    uninstall()
    assert all('forward' not in vars(layer.self_attn) for layer in module.encoder.layers)
    lib.torch.testing.assert_close(_forecast(lib, module, context, 64), expected, rtol=0, atol=0)
    monkeypatch.setattr(lib.attention, 'upstream_forward_digest', lambda: 'changed')
    with pytest.raises(RuntimeError, match='revalidate'):
        lib.attention.install_univariate_attention(module)


def test_runtime_installs_only_when_selected(lib, monkeypatch):
    runtime = pytest.importorskip('vtsfm.adapters.moirai2.runtime')
    monkeypatch.delenv(MOIRAI_ATTENTION_ENV, raising=False)
    default = runtime.MoiraiRuntime(TINY)
    assert default.attention_mode == 'upstream'
    assert all('forward' not in vars(layer.self_attn) for layer in default.model.encoder.layers)
    monkeypatch.setenv(MOIRAI_ATTENTION_ENV, 'univariate')
    selected = runtime.MoiraiRuntime(TINY)
    assert selected.attention_mode == 'univariate'
    assert all('forward' in vars(layer.self_attn) for layer in selected.model.encoder.layers)
    monkeypatch.setenv(MOIRAI_ATTENTION_ENV, 'bogus')
    with pytest.raises(ValueError):
        runtime.MoiraiRuntime(TINY)


def test_runtime_pins_univariate_attention_to_fp32(lib, monkeypatch):
    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime

    monkeypatch.setenv(MOIRAI_ATTENTION_ENV, 'univariate')
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')
    monkeypatch.delenv('VTSFM_PRECISION', raising=False)
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE', raising=False)
    torch = lib.torch
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        MoiraiRuntime(TINY)
        assert torch.get_float32_matmul_precision() == 'highest'
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.set_float32_matmul_precision(previous)


def test_snapshot_creates_its_output_directory_before_gpu_work(tmp_path, monkeypatch):
    from types import SimpleNamespace as Args

    from benchmarks import moirai_attention as bench

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(bench, '_setup', stop)
    monkeypatch.setitem(__import__('sys').modules, 'torch', __import__('types').ModuleType('torch'))
    monkeypatch.setitem(__import__('sys').modules, 'pickle', __import__('pickle'))
    out = tmp_path / 'new' / 'results'
    with pytest.raises(Stop):
        bench.snapshot(Args(output=str(out), batch=1, context=32, horizon=16))
    assert out.is_dir()
