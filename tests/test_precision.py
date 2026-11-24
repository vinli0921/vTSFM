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


def test_explicit_precision_overrides_the_environment_only_in_the_child(monkeypatch):
    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '1')
    command, environment = build_vllm_command(adapter='toto2', forecast_precision='fp32')
    assert environment[PRECISION_ENV] == 'fp32'
    assert environment[TF32_OVERRIDE_ENV] == '0'
    assert os.environ[TF32_OVERRIDE_ENV] == '1'
    assert command[command.index('--dtype') + 1] == 'float32'


def test_cli_precision_environment_matches_explicit_selection(monkeypatch):
    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    _, inherited = build_vllm_command(adapter='moirai2')
    _, explicit = build_vllm_command(adapter='moirai2', forecast_precision='bf16')
    assert inherited == explicit


def test_default_launch_preserves_existing_cuda_math_settings(monkeypatch):
    monkeypatch.delenv(PRECISION_ENV, raising=False)
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '1')
    _, environment = build_vllm_command(adapter='toto2')
    assert PRECISION_ENV not in environment
    assert environment[TF32_OVERRIDE_ENV] == '1'


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
    ('toto2', 'autocast'),
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


def test_continuous_fp32_transport_keeps_explicit_tf32_selection_available():
    _, environment = build_vllm_command(adapter='timesfm25', continuous_batching=True,
                                         forecast_precision='tf32')
    assert environment[CONTINUOUS_ENV] == '1'
    assert environment[PRECISION_ENV] == 'tf32'


def test_invalid_worker_mode_does_not_import_torch(monkeypatch):
    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    with pytest.raises(ValueError, match='Toto and Moirai'):
        configure_precision('timesfm25')


def test_unspecified_worker_precision_preserves_existing_torch_settings(monkeypatch):
    monkeypatch.delenv(PRECISION_ENV, raising=False)
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '1')
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert configure_precision('toto2') == 'fp32'


@pytest.mark.parametrize('precision,override', [(None, '0'), ('fp32', '0'), ('tf32', '1'), ('bf16', '0')])
def test_profile_child_environment_preserves_selected_precision(monkeypatch, precision, override):
    from model_serving_support import child_environment

    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    if precision is None:
        monkeypatch.delenv('VTSFM_TEST_PRECISION', raising=False)
    else:
        monkeypatch.setenv('VTSFM_TEST_PRECISION', precision)
    environment = child_environment(2)
    assert PRECISION_ENV not in environment
    assert environment[TF32_OVERRIDE_ENV] == override


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


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('rows', [1, 2])
def test_toto_cropped_output_owns_compact_fp32_storage(dtype, rows):
    torch = pytest.importorskip('torch')
    pytest.importorskip('toto2')
    from vtsfm.adapters.toto2.runtime import TotoRuntime

    horizon = 33
    if rows == 1:
        # Contiguous view into a still-larger backing tensor.
        backing = torch.arange(9 * horizon + 8, dtype=getattr(torch, dtype))
        native = backing[4:4 + 9 * horizon].reshape(9, rows, 1, horizon)
    else:
        backing = torch.arange(9 * rows * 64, dtype=getattr(torch, dtype))
        native = backing.reshape(9, rows, 1, 64)[..., :horizon]
    runtime = TotoRuntime.__new__(TotoRuntime)
    runtime.model = SimpleNamespace(config=SimpleNamespace(patch_size=32),
                                    forecast=lambda *args, **kwargs: native)
    context = torch.zeros(rows, 64)
    actual = runtime._forecast(context, torch.isfinite(context), horizon)
    torch.testing.assert_close(actual, native[:, :, 0].permute(1, 0, 2).float(), rtol=0, atol=0)
    assert actual.is_contiguous() and actual.storage_offset() == 0
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    assert actual.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()


@pytest.mark.gpu
@pytest.mark.parametrize('precision', ['fp32', 'bf16', 'tf32'])
def test_native_precision_rotary_tables_match_cpu_construction_on_cuda(precision, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('toto2')
    if not torch.cuda.is_available():
        pytest.skip('Requires CUDA for constructor-device parity')
    from dd_unit_scaling import RMSNorm
    from toto2 import Toto2Model
    from toto2.configuration import Toto2ModelConfig
    from toto2.model import RotaryProjection
    from vtsfm.adapters.toto2.precision import prepare_native_bf16
    from vtsfm.adapters.toto2.runtime import TotoRuntime

    if precision == 'fp32':
        monkeypatch.delenv(PRECISION_ENV, raising=False)
    else:
        monkeypatch.setenv(PRECISION_ENV, precision)
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '1' if precision == 'tf32' else '0')
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')

    config = dict(patch_size=32, d_model=16, num_heads=2, num_layers=2,
        layer_group_size=2, num_variate_layers_per_group=1, variate_layer_first=False,
        residual_attn_ratio=0.5, norm_include_weight=True)
    with torch.device('cpu'):
        native = Toto2Model(Toto2ModelConfig(**config)).eval()
    with torch.device('cuda:0'):
        served = TotoRuntime(config).model.eval()
    original = {name: {key: getattr(module, key).clone() for key in ('theta', 'cos', 'sin')}
                for name, module in native.named_modules() if isinstance(module, RotaryProjection)}
    native.to(device='cuda:0')
    if precision == 'bf16':
        prepare_native_bf16(native, RMSNorm, RotaryProjection)
    assert original
    for model in (native, served):
        for name, module in model.named_modules():
            if name in original:
                for key, expected in original[name].items():
                    actual = getattr(module, key)
                    assert actual.device.type == 'cuda' and actual.dtype == torch.float32
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('device,dtype,batch,sequence,fallback', [
    ('cuda', 'bfloat16', 65535, 1, False), ('cuda', 'bfloat16', 65536, 1, True),
    ('cuda', 'float32', 65536, 1, False), ('cpu', 'bfloat16', 65536, 1, False),
    ('cuda', 'bfloat16', 65536, 2, False),
])
def test_oversized_variate_backend_scope_preserves_calls_and_flags(device, dtype, batch, sequence, fallback):
    torch = pytest.importorskip('torch')
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from types import MethodType
    from vtsfm.adapters.toto2.precision import _large_variate_forward

    def flags():
        return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled())

    class Attention(torch.nn.Module):
        def forward(self, state, seq_ids=None, **kwargs):
            self.seen = flags(), seq_ids, kwargs['mask']
            if kwargs.get('fail'):
                raise RuntimeError('Original forward failed')
            return state

    attention = Attention()
    attention._vtsfm_native_attention_forward = attention.forward
    attention.forward = MethodType(_large_variate_forward, attention)
    state = SimpleNamespace(device=torch.device(device), dtype=getattr(torch, dtype), shape=(batch, sequence, 8))
    marker = object()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        before = flags()
        assert attention(state, seq_ids=marker, mask=marker) is state
        assert attention.seen == ((False, True, True, False) if fallback else before, marker, marker)
        assert flags() == before
        with pytest.raises(RuntimeError, match='Original forward failed'):
            attention(state, mask=marker, fail=True)
        assert flags() == before


def test_bf16_attention_wrapper_is_instance_local_and_survives_deepcopy():
    from copy import deepcopy
    from types import MethodType
    torch, model, norm, rotary = tiny_toto()
    from vtsfm.adapters.toto2.precision import prepare_native_bf16

    class Attention(torch.nn.Module):
        def __init__(self, variate):
            super().__init__()
            self.is_variate_layer = variate
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, state):
            return self.weight + state

    model.variate, model.time = Attention(True), Attention(False)

    def existing_override(module, state):
        return 2 * module.weight + state

    model.variate.forward = MethodType(existing_override, model.variate)
    keys = tuple(model.state_dict())
    prepare_native_bf16(model, norm, rotary, attention_type=Attention)
    assert tuple(model.state_dict()) == keys
    assert 'forward' in vars(model.variate) and 'forward' not in vars(model.time)
    candidate = deepcopy(model)
    assert candidate.variate.forward.__self__ is candidate.variate
    assert candidate.variate._vtsfm_native_attention_forward.__self__ is candidate.variate
    with torch.no_grad():
        candidate.variate.weight.fill_(3)
    assert candidate.variate(torch.tensor(1.)).item() == 7
    assert model.variate(torch.tensor(1.)).item() == 3
