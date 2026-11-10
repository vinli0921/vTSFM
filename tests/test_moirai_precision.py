"""Native BF16 boundaries against a small official Moirai model, without weights."""

import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('selected,expected', [(None, 'fp32'), ('bf16', 'bf16'), ('tf32', 'tf32')])
def test_native_reference_precision_ignores_ambient_server_selection(monkeypatch, selected, expected):
    import model_integration_support as support
    from vtsfm import precision

    class SelectionReached(Exception):
        pass

    calls = []
    def configure(adapter, mode):
        calls.append((adapter, mode))
        raise SelectionReached
    monkeypatch.setattr(precision, 'configure_precision', configure)
    monkeypatch.setattr(support.platform, 'system', lambda: 'Linux')
    monkeypatch.setenv(precision.PRECISION_ENV, 'bf16')
    if selected is None:
        monkeypatch.delenv('VTSFM_TEST_PRECISION', raising=False)
    else:
        monkeypatch.setenv('VTSFM_TEST_PRECISION', selected)
    # Stop at precision selection, before seeds, model imports or downloads.
    monkeypatch.setitem(sys.modules, 'numpy', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=None))
    with pytest.raises(SelectionReached):
        support.NativeForecaster('moirai2')
    assert calls == [('moirai2', expected)]


def test_tiny_moirai_preserves_fp32_numerics_and_recursive_quantiles(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('uni2ts')
    from uni2ts.module.norm import RMSNorm
    from uni2ts.module.position import RotaryProjection
    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime
    from vtsfm.precision import PRECISION_ENV, TF32_OVERRIDE_ENV

    monkeypatch.setenv(PRECISION_ENV, 'bf16')
    monkeypatch.setenv(TF32_OVERRIDE_ENV, '0')
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: pytest.fail('Native BF16 must not use autocast'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        runtime = MoiraiRuntime(dict(d_model=64, d_ff=128, num_layers=1,
            patch_size=16, num_predict_token=4, max_seq_len=64,
            attn_dropout_p=0.0, dropout_p=0.0, scaling=True))
    model = runtime.model.eval()
    assert model.in_proj.hidden_layer.weight.dtype == torch.bfloat16
    assert model.out_proj.output_layer.weight.dtype == torch.bfloat16
    for module in model.modules():
        if isinstance(module, RMSNorm):
            assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
        if isinstance(module, RotaryProjection):
            assert module.theta.dtype == module.cos.dtype == module.sin.dtype == torch.float32
    observed = []
    handle = model.in_proj.register_forward_pre_hook(lambda module, args: observed.append(args[0].dtype))
    original_quantile = torch.quantile

    def quantile(values, *args, **kwargs):
        assert values.dtype == torch.float32
        return original_quantile(values, *args, **kwargs)

    monkeypatch.setattr(torch, 'quantile', quantile)
    context = torch.sin(torch.arange(32, dtype=torch.float32) / 7).repeat(2, 1)
    short = runtime.forecast(context, 17)
    long = runtime.forecast(context, 129)
    repeated = runtime.forecast(context, 17)
    handle.remove()
    assert short.shape == (2, 9, 17) and long.shape == (2, 9, 129)
    assert short.dtype == long.dtype == torch.float32
    assert torch.isfinite(short).all() and torch.isfinite(long).all()
    torch.testing.assert_close(short, repeated, rtol=0, atol=0)
    assert observed and set(observed) == {torch.bfloat16}


def test_native_reference_preserves_bf16_through_ragged_recursive_predictions(monkeypatch):
    """Exercise the real native helper without checkpoint downloads or a GPU."""
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    pytest.importorskip('uni2ts')
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from model_integration_support import NativeForecaster, forecast_settings
    from vtsfm.adapters.moirai2.precision import prepare_native_bf16

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = Moirai2Module(d_model=64, d_ff=128, num_layers=1,
            patch_size=16, num_predict_token=4, max_seq_len=64,
            attn_dropout_p=0.0, dropout_p=0.0, scaling=True).eval()
    prepare_native_bf16(model)
    # Inject only the constructor's checkpoint/device state; predict_array and
    # the official ragged-length grouping/forecast path execute unchanged.
    native = NativeForecaster.__new__(NativeForecaster)
    native.adapter, native.device = 'moirai2', torch.device('cpu')
    native.settings = forecast_settings('moirai2')
    native.reference, native.forecast_class = model, Moirai2Forecast
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    observed = []
    def inspect_projection(module, args):
        assert module.hidden_layer.weight.dtype == torch.bfloat16
        observed.append(args[0].dtype)
    handle = model.in_proj.register_forward_pre_hook(inspect_projection)
    histories = [np.sin(np.arange(length, dtype=np.float32) / 7).tolist() for length in (32, 47)]
    outputs = []
    try:
        for horizon in (17, 129, 17):
            data = {'inputs': [{'target': history} for history in histories],
                    'parameters': {'prediction_length': horizon}}
            result = native.predict_array(data)
            assert result.shape == (2, 9, horizon)
            assert result.dtype == np.float32 and np.isfinite(result).all()
            assert model.in_proj.hidden_layer.weight.dtype == torch.bfloat16
            assert model.out_proj.output_layer.weight.dtype == torch.bfloat16
            outputs.append(result)
    finally:
        handle.remove()
    np.testing.assert_array_equal(outputs[0], outputs[2])
    assert observed and set(observed) == {torch.bfloat16}


def test_native_and_serving_checkpoint_orders_keep_exact_fp32_norms():
    torch = pytest.importorskip('torch')
    pytest.importorskip('uni2ts')
    from uni2ts.model.moirai2 import Moirai2Module
    from uni2ts.module.norm import RMSNorm
    from vtsfm.adapters.moirai2.precision import prepare_native_bf16

    config = dict(d_model=64, d_ff=128, num_layers=1, patch_size=16,
                  num_predict_token=4, max_seq_len=64, attn_dropout_p=0.0,
                  dropout_p=0.0, scaling=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        original = Moirai2Module(**config).eval()
        served = Moirai2Module(**config).eval()
        native = Moirai2Module(**config).eval()
    norm_names = set()
    with torch.no_grad():
        for path, module in original.named_modules():
            if isinstance(module, RMSNorm):
                for name, parameter in module.named_parameters(recurse=False):
                    norm_names.add(f'{path}.{name}')
                    # Unit-valued initialized norms would conceal BF16 rounding.
                    parameter.copy_(torch.linspace(0.1234567, 0.9876543,
                                                    parameter.numel()).reshape_as(parameter))
    checkpoint = {name: value.detach().clone() for name, value in original.state_dict().items()}
    tables = {name: value.clone() for name, value in original.named_buffers() if value.is_floating_point()}
    assert norm_names
    assert any(not torch.equal(checkpoint[name], checkpoint[name].bfloat16().float()) for name in norm_names)
    # Serving prepares before strict checkpoint loading; native reference loads
    # before preparing. Both must preserve the same unrounded FP32 norm values.
    prepare_native_bf16(served)
    served.load_state_dict(checkpoint, strict=True)
    native.load_state_dict(checkpoint, strict=True)
    prepare_native_bf16(native)
    for model in (served, native):
        model.to(device=torch.device('cpu'))
        assert tuple(model.state_dict()) == tuple(checkpoint)
        for name, parameter in model.named_parameters():
            expected = checkpoint[name] if name in norm_names else checkpoint[name].bfloat16()
            assert parameter.dtype == expected.dtype
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        for name, value in model.named_buffers():
            if name in tables:
                assert value.dtype == tables[name].dtype == torch.float32
                torch.testing.assert_close(value, tables[name], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('precision', ['bf16', 'tf32'])
def test_moirai_native_precision_rotary_tables_match_cpu_construction_on_cuda(precision, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('uni2ts')
    if not torch.cuda.is_available():
        pytest.skip('Requires CUDA for constructor-device parity')
    from uni2ts.model.moirai2 import Moirai2Module
    from uni2ts.module.position import RotaryProjection
    from vtsfm.adapters.moirai2.precision import prepare_native_bf16
    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime

    monkeypatch.setenv('VTSFM_PRECISION', precision)
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0' if precision == 'bf16' else '1')
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')

    config = dict(d_model=64, d_ff=128, num_layers=1, patch_size=16,
                  num_predict_token=4, max_seq_len=64, attn_dropout_p=0.0,
                  dropout_p=0.0, scaling=True)
    with torch.device('cpu'):
        native = Moirai2Module(**config).eval()
    with torch.device('cuda:0'):
        served = MoiraiRuntime(config).model.eval()
    original = {name: {key: getattr(module, key).clone() for key in ('theta', 'cos', 'sin')}
                for name, module in native.named_modules() if isinstance(module, RotaryProjection)}
    native.to(device='cuda:0')
    if precision == 'bf16':
        prepare_native_bf16(native)
    assert original
    for model in (native, served):
        for name, module in model.named_modules():
            if name in original:
                for key, expected in original[name].items():
                    actual = getattr(module, key)
                    assert actual.device.type == 'cuda' and actual.dtype == torch.float32
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('precision', ['bf16', 'tf32'])
def test_reduced_precision_ragged_serving_preserves_native_batch_geometry(monkeypatch, precision):
    """CPU outputs can hide CUDA rounding changes, so compare decoder inputs too."""
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    pytest.importorskip('uni2ts')
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from model_integration_support import NativeForecaster, forecast_settings, integration_cases
    from test_moirai_runtime import prepare, plan
    from vtsfm.adapters.moirai2.precision import prepare_native_bf16
    from vtsfm.adapters.moirai2.runtime import MoiraiRuntime

    monkeypatch.setenv('VTSFM_PRECISION', precision)
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0' if precision == 'bf16' else '1')
    config = dict(d_model=64, d_ff=128, num_layers=1, patch_size=16,
                  num_predict_token=4, max_seq_len=64, attn_dropout_p=0.0,
                  dropout_p=0.0, scaling=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        reference = Moirai2Module(**config).eval()
        runtime = MoiraiRuntime(config)
    if precision == 'bf16':
        prepare_native_bf16(reference)
    runtime.model.load_state_dict(reference.state_dict(), strict=True)
    runtime.model.eval()
    native = NativeForecaster.__new__(NativeForecaster)
    native.adapter, native.device = 'moirai2', torch.device('cpu')
    native.settings = forecast_settings('moirai2')
    native.reference, native.forecast_class = reference, Moirai2Forecast
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    source = integration_cases('moirai2')[1]['data']['inputs']
    histories = [np.asarray(source[index]['target'], dtype=np.float32) for index in (0, 1, 0)]
    native_inputs, serving_inputs = [], []
    handles = [reference.in_proj.register_forward_pre_hook(
                   lambda module, args: native_inputs.append(args[0].clone())),
               runtime.model.in_proj.register_forward_pre_hook(
                   lambda module, args: serving_inputs.append(args[0].clone()))]
    try:
        for order in (histories, [histories[1], histories[0], histories[2]]):
            results = []
            for horizon in (33, 160, 33):
                data = {'inputs': [{'target': history.tolist()} for history in order],
                        'parameters': {'prediction_length': horizon}}
                native_inputs.clear()
                serving_inputs.clear()
                expected = torch.as_tensor(native.predict_array(data))
                context = torch.full((len(order), 192), float('nan'))
                for row, history in zip(context, order):
                    row[-len(history):] = torch.as_tensor(history)
                original = context.clone()
                actual = runtime.forecast(context, horizon)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(context, original, rtol=0, atol=0, equal_nan=True)
                assert len(serving_inputs) == len(native_inputs) > 0
                for observed, wanted in zip(serving_inputs, native_inputs):
                    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
                results.append(actual)
                # Prepared execution uses finite padding and its CPU length
                # metadata; it must preserve the same native groups and order.
                serving_inputs.clear()
                request = prepare(SimpleNamespace(device=torch.device('cpu')), 'bf16',
                                  order, horizon, width=192)
                before = request.context.clone()
                executed = runtime.execute(plan(request)).values
                torch.testing.assert_close(executed, expected, rtol=0, atol=0)
                torch.testing.assert_close(request.context, before, rtol=0, atol=0)
                assert len(serving_inputs) == len(native_inputs)
                for observed, wanted in zip(serving_inputs, native_inputs):
                    torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
            torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()
