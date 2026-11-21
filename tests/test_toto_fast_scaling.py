"""Toto's inference-only unit-scaling shims (#74) on a tiny random CPU model.

Run in an environment with toto2 installed (skipped otherwise). These check
equality with the upstream autograd wrappers on CPU, not CUDA parity.
"""

import pytest

TINY = dict(patch_size=16, d_model=32, num_heads=2, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False, residual_attn_ratio=1.0)


@pytest.fixture
def toto(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('toto2')
    from vtsfm.adapters.toto2 import fast_scaling
    from vtsfm.adapters.toto2.runtime import TotoRuntime

    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')
    monkeypatch.delenv('VTSFM_TOTO_FAST_UNIT_SCALING', raising=False)
    torch.manual_seed(5)
    runtime = TotoRuntime(TINY)
    runtime.model.eval()
    yield runtime
    fast_scaling.uninstall()


def histories():
    import torch

    context = torch.full((3, 80), float('nan'))
    context[0] = torch.sin(torch.arange(80) / 7)
    context[1, 23:] = torch.cos(torch.arange(57) / 5) + 4
    context[2, 40:] = torch.linspace(-2, 3, 40)
    context[2, 50] = float('nan')  # A missing observation inside the history.
    return context


def count_function_calls(monkeypatch):
    """Count forwards of both libraries' autograd Functions (originals, not shims)."""
    import unit_scaling.scale

    from vtsfm.adapters.toto2 import fast_scaling

    calls = []

    def counting(function, forward):
        def counted(*args):  # Function.apply binds this signature.
            calls.append(function)
            return forward(*args)
        return staticmethod(counted)

    for function in (fast_scaling._InferenceScaledGrad.original, unit_scaling.scale._ScaledGrad):
        monkeypatch.setattr(function, 'forward', counting(function, function.forward))
    return calls


def test_installs_both_pinned_libraries(toto):
    import dd_unit_scaling.scale
    import unit_scaling.scale

    from vtsfm.adapters.toto2 import fast_scaling

    assert toto.fast_unit_scaling
    assert set(fast_scaling.install()) == {'dd_unit_scaling.scale', 'unit_scaling.scale'}
    assert dd_unit_scaling.scale._ScaledGrad is fast_scaling._InferenceScaledGrad
    assert unit_scaling.scale._scale.__wrapped__ is fast_scaling._installed['unit_scaling.scale'][1]


@pytest.mark.parametrize('horizon', [17, 800])
def test_fast_forecast_equals_upstream_wrappers(toto, horizon):
    import torch

    context = histories()
    fast = toto.forecast(context, horizon)
    toto.fast_unit_scaling = False
    native = toto.forecast(context, horizon)
    assert fast.shape == (3, 9, horizon)
    torch.testing.assert_close(fast, native, rtol=0, atol=0)


def test_forecast_skips_autograd_functions_only_inside_the_runtime(toto, monkeypatch):
    import torch

    calls = count_function_calls(monkeypatch)
    toto.forecast(histories(), 17)
    assert calls == []

    # An official model in the same process keeps the upstream wrappers.
    inputs = {'target': torch.zeros((1, 1, 32)), 'target_mask': torch.ones((1, 1, 32), dtype=torch.bool),
              'series_ids': torch.zeros((1, 1), dtype=torch.long)}
    with torch.inference_mode():
        toto.model.forecast(inputs, horizon=17, decode_block_size=768)
    assert calls


def test_grad_enabled_keeps_upstream_backward_scaling(toto):
    import torch
    from dd_unit_scaling import functional as U

    from vtsfm.adapters.toto2.fast_scaling import fast_unit_scaling

    x = torch.ones(4, requires_grad=True)
    with fast_unit_scaling():
        residual, skip = U.residual_split(x, tau=0.5)
        (residual.sum() + 2 * skip.sum()).backward()
    denominator = (1 + 0.25) ** 0.5
    torch.testing.assert_close(x.grad, torch.full((4,), 0.5 / denominator + 2 / denominator))


def test_opt_out_and_source_change_keep_native_wrappers(monkeypatch, caplog):
    pytest.importorskip('toto2')
    import dd_unit_scaling.scale as dd_scale

    from vtsfm.adapters.toto2 import fast_scaling
    from vtsfm.adapters.toto2.runtime import TotoRuntime

    original = dd_scale._ScaledGrad
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')
    monkeypatch.setenv('VTSFM_TOTO_FAST_UNIT_SCALING', '0')
    assert not TotoRuntime(TINY).fast_unit_scaling
    assert dd_scale._ScaledGrad is original

    monkeypatch.setitem(fast_scaling.PINNED_SCALE_SOURCES, 'dd_unit_scaling.scale', '0' * 64)
    try:
        with caplog.at_level('WARNING'):
            assert fast_scaling.install() == ('unit_scaling.scale',)
        assert dd_scale._ScaledGrad is original
        assert 'dd_unit_scaling.scale source changed' in caplog.text
    finally:
        fast_scaling.uninstall()
