"""Explicit precision selection and native dtype boundaries, without checkpoints."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vtsfm.cli import build_vllm_command
from vtsfm.precision import PRECISION_ENV, TF32_OVERRIDE_ENV, configure_precision
from vtsfm.stepping import CONTINUOUS_ENV


def test_default_does_not_inherit_an_experimental_worker_precision(monkeypatch):
    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '1')
    command, environment = build_vllm_command(adapter='toto2')
    assert environment[PRECISION_ENV] == 'fp32'
    assert environment[TF32_OVERRIDE_ENV] == '0'
    assert os.environ[TF32_OVERRIDE_ENV] == '1'
    assert command[command.index('--dtype') + 1] == 'float32'


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_tf32_is_explicit_and_keeps_the_fp32_transport(adapter, monkeypatch):
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '0')
    command, environment = build_vllm_command(adapter=adapter, forecast_precision='tf32')
    assert environment[PRECISION_ENV] == 'tf32'
    assert environment[TF32_OVERRIDE_ENV] == '1'
    assert command[command.index('--dtype') + 1] == 'float32'


def test_toto_bf16_changes_compute_selection_not_the_wire_dtype():
    command, environment = build_vllm_command(adapter='toto2', forecast_precision='bf16')
    assert environment[PRECISION_ENV] == 'bf16'
    assert environment[TF32_OVERRIDE_ENV] == '0'
    assert command[command.index('--dtype') + 1] == 'float32'


@pytest.mark.parametrize('adapter,precision', [
    ('chronos2', 'tf32'), ('chronos2', 'bf16'), ('timesfm25', 'bf16'),
    ('moirai2', 'bf16'), ('toto2', 'autocast'),
])
def test_unsupported_modes_fail_before_starting_a_worker(adapter, precision):
    with pytest.raises(ValueError):
        build_vllm_command(adapter=adapter, forecast_precision=precision)


@pytest.mark.parametrize('precision,setting,tf32', [
    ('fp32', 'highest', False), ('tf32', 'high', True), ('bf16', 'highest', False),
])
def test_worker_sets_the_requested_math_policy(monkeypatch, precision, setting, tf32):
    monkeypatch.delenv(TF32_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(CONTINUOUS_ENV, raising=False)
    calls = []
    torch = SimpleNamespace(set_float32_matmul_precision=calls.append,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None))))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setenv(PRECISION_ENV, precision)
    assert configure_precision('toto2') == precision
    assert calls == [setting]
    assert torch.backends.cuda.matmul.allow_tf32 is tf32


@pytest.mark.parametrize('precision,override', [
    ('fp32', '1'), ('tf32', '0'), ('bf16', '1'), ('tf32', 'invalid'),
])
def test_direct_worker_rejects_conflicting_cuda_overrides(monkeypatch, precision, override):
    monkeypatch.setenv(PRECISION_ENV, precision)
    monkeypatch.setenv(TF32_OVERRIDE_ENV, override)
    with pytest.raises(ValueError, match=TF32_OVERRIDE_ENV):
        configure_precision('toto2')


def test_chronos_keeps_its_existing_cuda_override(monkeypatch):
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '1')
    _, environment = build_vllm_command(adapter='chronos2')
    assert environment[TF32_OVERRIDE_ENV] == '1'


def test_continuous_mode_rejects_unqualified_precision_in_both_entry_points(monkeypatch):
    with pytest.raises(ValueError, match='requires FP32'):
        build_vllm_command(adapter='timesfm25', continuous_batching=True, forecast_precision='tf32')
    monkeypatch.setenv(PRECISION_ENV, 'tf32')
    monkeypatch.setenv(CONTINUOUS_ENV, '1')
    with pytest.raises(ValueError, match='requires FP32'):
        configure_precision('timesfm25')


def test_invalid_worker_mode_does_not_import_torch(monkeypatch):
    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    with pytest.raises(ValueError, match='Toto only'):
        configure_precision('moirai2')


def test_dry_run_and_precision_discovery_stay_cpu_safe():
    code = '''
import sys
from vtsfm.cli import main
from vtsfm.precision import validate_precision
assert validate_precision('toto2', 'bf16') == 'bf16'
assert main(['serve', '--adapter', 'toto2', '--forecast-precision', 'bf16', '--dry-run']) == 0
assert not {'torch', 'vllm', 'toto2', 'uni2ts', 'timesfm'}.intersection(sys.modules)
'''
    process = subprocess.run([sys.executable, '-c', code], env=os.environ.copy(),
                             capture_output=True, text=True, check=True)
    payload = json.loads(process.stdout)
    assert payload['environment'][PRECISION_ENV] == 'bf16'


def tiny_toto():
    torch = pytest.importorskip('torch')

    class Norm(torch.nn.RMSNorm):
        def forward(self, values):
            self.seen_dtype = values.dtype
            return super().forward(values)

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('table', torch.linspace(0.1234567, 0.9876543, 8), persistent=False)

        def forward(self, values):
            return values * self.table.to(values.dtype)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_proj = torch.nn.Linear(4, 8)
            self.norm = Norm(8)
            self.rotary = Rotary()
            self.output_head = torch.nn.Linear(8, 9)
            self.register_buffer('residual_scale', torch.tensor(0.2345678))
            self._kv_cache = None
            with torch.no_grad():
                self.norm.weight.fill_(0.1234567)

        def forward(self, context):
            assert context.dtype == torch.float32
            # The official scaler/forecast buffers stay FP32; it computes
            # statistics in FP64 and returns loc/scale in the input dtype.
            loc = context.double().mean(-1, keepdim=True).float()
            scale = context.double().std(-1, keepdim=True).float().clamp_min(1e-6)
            state = self.patch_proj(((context - loc) / scale).asinh())
            state = self.norm(self.rotary(state))
            self.state_dtype = state.dtype
            quantiles = self.output_head(state)
            self.head_dtype = quantiles.dtype
            return quantiles.sinh() * scale + loc

    return torch, Model().eval(), Norm, Rotary


def test_native_bf16_keeps_sensitive_values_and_checkpoint_names(monkeypatch):
    torch, model, norm, rotary = tiny_toto()
    from vtsfm.adapters.toto2.precision import prepare_native_bf16

    names = tuple(model.state_dict())
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    weight, table = model.norm.weight.detach().clone(), model.rotary.table.clone()
    residual_scale = model.residual_scale.clone()
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: pytest.fail('Native BF16 must not use autocast'))
    handles = prepare_native_bf16(model, norm, rotary)
    assert tuple(model.state_dict()) == names
    assert model.patch_proj.weight.dtype == model.output_head.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == model.rotary.table.dtype == torch.float32
    assert torch.equal(model.norm.weight, weight) and torch.equal(model.rotary.table, table)
    assert model.residual_scale.dtype == torch.float32
    assert torch.equal(model.residual_scale, residual_scale)
    loaded = model.load_state_dict(checkpoint, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert model.patch_proj.weight.dtype == torch.bfloat16
    output = model(torch.tensor([[1., 2., 3., 4.], [2., 3., 4., 5.]]))
    assert model.norm.seen_dtype == torch.float32
    assert model.state_dtype == torch.bfloat16 and model.head_dtype == torch.float32
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    assert handles


def test_precision_hooks_are_instance_local_and_cannot_be_installed_twice():
    torch, model, norm, rotary = tiny_toto()
    from vtsfm.adapters.toto2.precision import prepare_native_bf16

    _, other, _, _ = tiny_toto()
    original = other(torch.tensor([[1., 2., 3., 4.]]))
    prepare_native_bf16(model, norm, rotary)
    assert other.patch_proj.weight.dtype == torch.float32
    torch.testing.assert_close(other(torch.tensor([[1., 2., 3., 4.]])), original, rtol=0, atol=0)
    with pytest.raises(ValueError, match='already configured'):
        prepare_native_bf16(model, norm, rotary)


def test_cached_forecast_rejects_late_dtype_changes_without_mutating_weights():
    torch, model, norm, rotary = tiny_toto()
    from vtsfm.adapters.toto2.precision import prepare_native_bf16

    model._kv_cache = object()
    with pytest.raises(ValueError, match='before its first'):
        prepare_native_bf16(model, norm, rotary)
    assert model.patch_proj.weight.dtype == torch.float32


def test_tiny_official_toto_runs_native_bf16_without_autocast(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('toto2')
    from vtsfm.adapters.toto2.runtime import TotoRuntime

    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '0')
    monkeypatch.delenv(CONTINUOUS_ENV, raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        runtime = TotoRuntime(dict(patch_size=32, d_model=16, num_heads=2, num_layers=2,
            layer_group_size=2, num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=0.5, norm_include_weight=True))
    runtime.model.eval()
    context = torch.sin(torch.arange(64, dtype=torch.float32) / 7).repeat(2, 1)
    original = context.clone()
    forecasts = []
    for horizon in (96, 800, 96):
        result = runtime.forecast(context, horizon)
        forecasts.append(result)
        assert result.shape == (2, 9, horizon)
        assert result.dtype == torch.float32 and torch.isfinite(result).all()
    torch.testing.assert_close(context, original, rtol=0, atol=0)
    torch.testing.assert_close(forecasts[0], forecasts[2], rtol=0, atol=0)
    assert runtime.model.patch_proj.linear1.weight.dtype == torch.bfloat16
