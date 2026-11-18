"""Tiny official CPU models: real compiler routing, forecasts and checkpoint keys.

No weights/downloads/GPU are used. Run inside each model environment. These
tests exercise Inductor on small random models, not current-stack CUDA parity.
"""

from copy import deepcopy
from dataclasses import replace
import importlib

import pytest


@pytest.mark.parametrize('adapter', ['timesfm25', 'toto2', 'moirai2'])
def test_compiled_tiny_forecasts_preserve_state_keys_inputs_and_retained_shapes(adapter, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip({'timesfm25': 'timesfm', 'toto2': 'toto2', 'moirai2': 'uni2ts'}[adapter])
    from torch._dynamo import config
    from torch._dynamo.utils import counters

    torch.set_num_threads(2)
    torch.manual_seed(3)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION', '0')
    monkeypatch.setenv('VTSFM_FORECAST_EXECUTION', 'eager')
    module = importlib.import_module(f'vtsfm.adapters.{adapter}.runtime')
    if adapter == 'timesfm25':
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        definition = TimesFM_2p5_200M_torch_module.config
        tiny = replace(definition,
            tokenizer=replace(definition.tokenizer, hidden_dims=32, output_dims=32),
            stacked_transformers=replace(definition.stacked_transformers, num_layers=1,
                transformer=replace(definition.stacked_transformers.transformer,
                                    model_dims=32, hidden_dims=64, num_heads=2)),
            output_projection_point=replace(definition.output_projection_point, input_dims=32, hidden_dims=32),
            output_projection_quantiles=replace(definition.output_projection_quantiles, input_dims=32, hidden_dims=32))
        monkeypatch.setattr(TimesFM_2p5_200M_torch_module, 'config', tiny)
        runtime = module.TimesFMRuntime({})
        horizon = 129
    elif adapter == 'toto2':
        runtime = module.TotoRuntime(dict(patch_size=16, d_model=32, num_heads=2,
            num_layers=2, layer_group_size=2, num_variate_layers_per_group=1,
            variate_layer_first=False, residual_attn_ratio=1.0))
        horizon = 800
    else:
        runtime = module.MoiraiRuntime(dict(d_model=64, d_ff=128, num_layers=1,
            patch_size=16, max_seq_len=128, num_predict_token=4,
            attn_dropout_p=0.0, dropout_p=0.0))
        horizon = 129
    runtime.model.eval()
    candidate = deepcopy(runtime)
    keys = tuple(candidate.model.state_dict())
    parameters = {name: value for name, value in candidate.model.named_parameters()}
    if adapter == 'toto2':
        layer_class = type(runtime.model.transformer.layers[0])
        upstream_forward = layer_class.forward
    a = torch.sin(torch.arange(33, dtype=torch.float32) / 7).unsqueeze(0)
    b = torch.cos(torch.arange(128, dtype=torch.float32).reshape(2, 64) / 11)
    b[0, :33] = float('nan')  # A masked, unequal history remains independent.
    workloads = [(a, 17), (b, horizon), (b + 3, horizon), (a, 17)]
    expected = [runtime.forecast(context, length) for context, length in workloads]
    before = counters['stats']['unique_graphs']
    # The helper deliberately owns these settings for a worker's lifetime.
    # This test process shares unrelated tests, so restore its configuration.
    previous = {name: getattr(config, name) for name in
                ('recompile_limit', 'accumulated_recompile_limit',
                 'fail_on_recompile_limit_hit', 'suppress_errors')}
    try:
        module.compile_model(candidate.model)
        if adapter == 'toto2':
            assert layer_class.forward is upstream_forward
            assert all('forward' not in vars(layer) for layer in runtime.model.transformer.layers)
            low_precision = any(parameter.dtype in (torch.bfloat16, torch.float16)
                                for parameter in candidate.model.transformer.parameters())
            assert all(('forward' in vars(layer)) == low_precision
                       for layer in candidate.model.transformer.layers)
        assert tuple(candidate.model.state_dict()) == keys
        assert all(dict(candidate.model.named_parameters())[name] is value for name, value in parameters.items())
        candidate.model.load_state_dict(runtime.model.state_dict(), strict=True)
        for (context, length), reference in zip(workloads, expected):
            original = context.clone()
            actual = candidate.forecast(context, length)
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(context, original, rtol=0, atol=0, equal_nan=True)
        assert counters['stats']['unique_graphs'] > before, 'No compiler graph was executed'
        # Compiling one instance must not mutate the reference or class methods.
        core = {'timesfm25': runtime.model, 'toto2': getattr(runtime.model, 'transformer', None),
                'moirai2': getattr(runtime.model, 'encoder', None)}[adapter]
        assert 'forward' not in vars(core)
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def test_low_precision_toto_setup_failure_restores_instance_layer_forwards(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('toto2')
    from toto2 import Toto2Model
    from toto2.configuration import Toto2ModelConfig
    from vtsfm.adapters.toto2 import runtime

    model = Toto2Model(Toto2ModelConfig(patch_size=16, d_model=32, num_heads=2,
        num_layers=2, layer_group_size=2, num_variate_layers_per_group=1,
        variate_layer_first=False, residual_attn_ratio=1.0)).to(torch.bfloat16)
    keys = tuple(model.state_dict())
    upstream_forward = type(model.transformer.layers[0]).forward

    def fail(*args, **kwargs):
        raise RuntimeError('compiler setup failed')

    monkeypatch.setattr(runtime, 'compile_core', fail)
    with pytest.raises(RuntimeError, match='compiler setup failed'):
        runtime.compile_model(model)
    assert type(model.transformer.layers[0]).forward is upstream_forward
    assert all('forward' not in vars(layer) for layer in model.transformer.layers)
    assert tuple(model.state_dict()) == keys
