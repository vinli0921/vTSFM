"""Native-only issue #47 checks; no checkpoint, CUDA, serving, or timing claims.

Run official tiny-model CPU checks in the pinned Moirai environment:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests .venv-moirai2/bin/python -m pytest \
    tests/test_moirai_multivariate_attention.py -q
"""

import copy
import gc
import importlib
import subprocess
import sys
import weakref

import pytest


RTOL = ATOL = 1e-4


def test_import_does_not_load_runtime_packages():
    code = ("import sys; import vtsfm.adapters.moirai2.multivariate_attention; "
            "assert 'torch' not in sys.modules; assert 'uni2ts' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_public_capabilities_remain_univariate():
    from vtsfm.adapters import get_adapter

    spec = get_adapter('moirai2')
    assert not spec.supports_group_attention
    assert not spec.execution_capabilities.supports_grouped_rows


@pytest.fixture(scope='module')
def native():
    torch = pytest.importorskip('torch')
    pytest.importorskip('uni2ts')
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from uni2ts.common.torch_util import packed_causal_attention_mask

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch, Moirai2Module, Moirai2Forecast, packed_causal_attention_mask
    torch.set_num_threads(previous_threads)


def tiny_model(native, *, heads=1, layers=2):
    torch, Module, _, _ = native
    torch.manual_seed(47)
    model = Module(d_model=64 * heads, d_ff=96, num_layers=layers, patch_size=4,
                   max_seq_len=128, attn_dropout_p=0.0, dropout_p=0.0,
                   num_predict_token=2).eval()
    # Head-distinct, nonconstant biases make a dropped/swapped bias detectable.
    with torch.no_grad():
        for layer in model.encoder.layers:
            weights = layer.self_attn.var_attn_bias.emb.weight
            weights[0] = torch.linspace(-1.9, -0.7, weights.shape[-1])
            weights[1] = torch.linspace(1.1, 2.3, weights.shape[-1])
    return model


def install(model):
    from vtsfm.adapters.moirai2.multivariate_attention import install_multivariate_attention
    return install_multivariate_attention(model)


def test_pinned_source_guard_and_atomic_rejection(native, monkeypatch):
    module = importlib.import_module('vtsfm.adapters.moirai2.multivariate_attention')
    model = tiny_model(native)
    assert module.upstream_source_digests() == module.PINNED_SOURCE_SHA256
    monkeypatch.setitem(module.PINNED_SOURCE_SHA256, 'GroupedQueryAttention.forward', 'changed')
    with pytest.raises(RuntimeError, match='source changed'):
        install(model)
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    assert all('forward' not in vars(layer.self_attn) for layer in model.encoder.layers)


def test_installer_checks_every_layer_before_mutating_and_restores(native):
    model = tiny_model(native)
    attention = model.encoder.layers[-1].self_attn
    attention.attn_dropout_p = 0.1
    with pytest.raises(ValueError, match='configuration'):
        install(model)
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    assert 'forward' not in vars(model.encoder.layers[0].self_attn)
    attention.attn_dropout_p = 0.0
    undo = install(model)
    assert len(model._forward_pre_hooks) == 1
    assert len(model._forward_hooks) == 1
    with pytest.raises(ValueError, match='configuration'):
        install(model)
    undo()
    undo()
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    assert all('forward' not in vars(layer.self_attn) for layer in model.encoder.layers)


def test_packed_multivariate_attention_and_drop_bias_negative_control(native, monkeypatch):
    torch, _, _, packed_mask = native
    reference = tiny_model(native, heads=2)
    candidate = copy.deepcopy(reference)
    attention = reference.encoder.layers[0].self_attn
    patched = candidate.encoder.layers[0].self_attn
    # Includes sample zero, independent packed samples, nonmonotonic/repeated
    # time IDs, and three variates. Token-index causality would be incorrect.
    samples = torch.tensor([[0, 0, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2, 2]])
    times = torch.tensor([[0, 1, 0, 2, 1, 0, 1], [2, 0, 1, 1, 0, 2, 1]])
    variates = torch.tensor([[0, 1, 0, 1, 2, 0, 1], [0, 1, 2, 0, 0, 1, 2]])
    mask = packed_mask(samples, times)
    assert mask[0, 0, 0] and mask[0, 1, 0]  # sample zero follows native equality.
    assert mask[0, 3, 4] and not mask[0, 4, 3]  # time, not token order.
    hidden = torch.randn(2, 7, 128)
    before = hidden.clone()
    kwargs = dict(attn_mask=mask, query_var_id=variates, kv_var_id=variates,
                  query_time_id=times, kv_time_id=times)
    with torch.inference_mode():
        expected = attention(hidden, hidden, hidden, **kwargs)
    undo = install(candidate)
    # Dense native bias/mask assembly must never run in the candidate.
    def forbidden(*args, **kwargs):
        raise AssertionError('dense per-head native attention bias was materialized')
    monkeypatch.setattr(patched.var_attn_bias, 'forward', forbidden)
    monkeypatch.setattr(patched, '_update_attn_mask', forbidden)
    with torch.inference_mode():
        actual = patched(hidden, hidden, hidden, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        patched.var_attn_bias.emb.weight.zero_()
        dropped = patched(hidden, hidden, hidden, **kwargs)
    assert not torch.allclose(dropped, expected, rtol=RTOL, atol=ATOL)
    assert (dropped - expected).abs().max() > 1e-2
    assert torch.equal(hidden, before)
    undo()


def forecast(native, model, target, observed, padding, horizon):
    _, _, Forecast, _ = native
    wrapper = Forecast(module=model, prediction_length=horizon, context_length=target.shape[1],
                       target_dim=target.shape[2], feat_dynamic_real_dim=0,
                       past_feat_dynamic_real_dim=0).to('cpu').eval()
    return wrapper(past_target=target, past_observed_target=observed, past_is_pad=padding)


@pytest.mark.parametrize('variables,horizon', [(3, 1), (3, 8), (1, 9), (1, 17)])
def test_native_ragged_forecast_and_univariate_recursion(native, variables, horizon):
    torch, _, _, _ = native
    reference = tiny_model(native)
    candidate = copy.deepcopy(reference)
    target = torch.randn(2, 19, variables)
    observed = torch.ones_like(target, dtype=torch.bool)
    padding = torch.zeros(2, 19, dtype=torch.bool)
    # Whole and partial left-padded patches plus interior missing observations
    # compare the exact same native inputs; no public ragged/imputation change.
    target[0, :7] = 0
    observed[0, :7] = False
    padding[0, :7] = True
    target[1, 9, 0] = 0
    observed[1, 9, 0] = False
    originals = [value.clone() for value in (target, observed, padding)]
    seen_batch_ranks = []
    hook = candidate.register_forward_pre_hook(lambda m, args: seen_batch_ranks.append(args[0].ndim))
    with torch.inference_mode():
        expected = forecast(native, reference, target, observed, padding, horizon)
        undo = install(candidate)
        actual = forecast(native, candidate, target, observed, padding, horizon)
    assert actual.shape == ((2, 9, horizon, variables) if variables > 1 else (2, 9, horizon))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    assert 3 in seen_batch_ranks
    if horizon > 8:
        assert 4 in seen_batch_ranks  # recursive nine-trajectory leading batch axis.
    for value, original in zip((target, observed, padding), originals):
        assert torch.equal(value, original)
    undo()
    hook.remove()


def test_recursive_native_float_noise_baseline_and_changed_input_aba(native):
    torch, _, _, _ = native
    reference = tiny_model(native)
    noisy = copy.deepcopy(reference)
    candidate = copy.deepcopy(reference)
    target_a = torch.randn(2, 19, 1)
    target_b = torch.randn(1, 23, 1) * 2 + 5

    def run(model, target):
        return forecast(native, model, target, torch.ones_like(target, dtype=torch.bool),
                        torch.zeros(target.shape[:2], dtype=torch.bool), 17)

    # One representable FP32 step at each native attention output is a separate
    # diagnostic of recursive noise propagation, never an alternative tolerance.
    def one_ulp(module, args, output):
        return torch.nextafter(output, torch.full_like(output, float('inf')))

    hooks = [layer.self_attn.register_forward_hook(one_ulp) for layer in noisy.encoder.layers]
    with torch.inference_mode():
        golden_a, golden_b = run(reference, target_a), run(reference, target_b)
        native_noise = run(noisy, target_a)
        undo = install(candidate)
        first_a, actual_b, final_a = run(candidate, target_a), run(candidate, target_b), run(candidate, target_a)
    assert (native_noise - golden_a).abs().max() > 0
    torch.testing.assert_close(native_noise, golden_a, rtol=RTOL, atol=ATOL)
    for actual, expected in [(first_a, golden_a), (actual_b, golden_b), (final_a, golden_a)]:
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    assert torch.equal(first_a, final_a)
    for hook in hooks:
        hook.remove()
    undo()


def test_pristine_upstream_multivariate_recursive_wrapper_limitation(native):
    from einops import EinopsError

    torch, _, _, _ = native
    # This is an upstream regression witness, not an accepted candidate failure.
    # Keep the native implementation pristine rather than repair the reference.
    model = tiny_model(native)
    target = torch.randn(2, 19, 3)
    with torch.inference_mode(), pytest.raises(EinopsError, match='Shape mismatch, 6 != 2'):
        forecast(native, model, target, torch.ones_like(target, dtype=torch.bool),
                 torch.zeros(2, 19, dtype=torch.bool), 17)


def test_pristine_checkpoint_geometry_multivariate_recursive_limitation(native):
    from einops import EinopsError

    torch, Module, _, _ = native
    # Literal config.json at Salesforce/moirai-2.0-R-small revision
    # 30f43ff08c8494f4943ae1521e9d4e94a0fbb389; weights are not needed to
    # reproduce the shape failure. No installer runs on this model.
    model = Module(d_model=384, d_ff=1024, num_layers=6, patch_size=16,
                   max_seq_len=512, attn_dropout_p=0, dropout_p=0,
                   scaling=True, num_predict_token=4).eval()
    target = torch.randn(1, 33, 2)
    assert not model._forward_pre_hooks
    assert all('forward' not in vars(layer.self_attn) for layer in model.encoder.layers)
    with torch.inference_mode(), pytest.raises(EinopsError, match='Shape mismatch, 8 != 4'):
        forecast(native, model, target, torch.ones_like(target, dtype=torch.bool),
                 torch.zeros(1, 33, dtype=torch.bool), 65)


def test_layers_share_one_flex_graph_and_one_block_mask_per_module_call(native, monkeypatch):
    """CPU Dynamo structure check; does not compile or prove a CUDA kernel."""
    torch, _, _, _ = native
    flex = importlib.import_module('torch.nn.attention.flex_attention')
    model = tiny_model(native, heads=2, layers=3)
    graphs, masks = [], []
    original_create = flex.create_block_mask

    def backend(graph, args):
        graphs.append(graph)
        return graph.forward

    def create(*args, **kwargs):
        result = original_create(*args, **kwargs)
        masks.append(result)
        return result

    monkeypatch.setattr(flex, 'create_block_mask', create)
    monkeypatch.setattr(flex, 'flex_attention', torch.compile(
        flex.flex_attention, backend=backend, fullgraph=True, dynamic=False))
    undo = install(model)
    target = torch.randn(2, 19, 3)
    observed, padding = torch.ones_like(target, dtype=torch.bool), torch.zeros(2, 19, dtype=torch.bool)
    with torch.inference_mode():
        first = forecast(native, model, target, observed, padding, 8)
        changed = forecast(native, model, target + 3, observed, padding, 8)
    assert len(graphs) == 1  # bias Parameters / IDs / mask are graph inputs.
    assert len(masks) == 2  # one mask for three layers in each native call.
    assert all(mask.kv_num_blocks.shape[1] == 1 for mask in masks)
    assert not torch.equal(first, changed)
    assert any(node.target is torch.ops.higher_order.flex_attention for node in graphs[0].graph.nodes)
    undo()


@pytest.mark.parametrize('fail', [False, True])
def test_native_call_releases_block_mask_and_captured_dense_mask(native, monkeypatch, fail):
    module = importlib.import_module('vtsfm.adapters.moirai2.multivariate_attention')
    torch, _, _, _ = native
    model = tiny_model(native)
    states, refs = [], []
    original_get = module._MaskState.get

    def get(state, mask, *args):
        block_mask = original_get(state, mask, *args)
        if not states:
            states.append(state)
            refs.extend([weakref.ref(mask), weakref.ref(block_mask)])
            # mask_mod closes over the reshaped quadratic boolean tensor.
            refs.extend(weakref.ref(cell.cell_contents) for cell in block_mask.mask_mod.__closure__
                        if isinstance(cell.cell_contents, torch.Tensor))
        return block_mask

    monkeypatch.setattr(module._MaskState, 'get', get)
    undo = install(model)
    if fail:
        def raise_after_attention(*args, **kwargs):
            raise RuntimeError('intentional output-projection failure')
        monkeypatch.setattr(model.encoder.layers[0].self_attn.out_proj, 'forward', raise_after_attention)
    target = torch.randn(1, 19, 3)

    def invoke():
        with torch.inference_mode():
            try:
                forecast(native, model, target, torch.ones_like(target, dtype=torch.bool),
                         torch.zeros(1, 19, dtype=torch.bool), 8)
            except RuntimeError as exc:
                if not fail or str(exc) != 'intentional output-projection failure':
                    raise
                return
            assert not fail, 'expected the intentional model failure'

    invoke()
    assert len(states) == 1 and len(refs) == 3
    assert states[0].ref is None and states[0].block_mask is None
    gc.collect()
    assert all(ref() is None for ref in refs)
    undo()


def test_multivariate_module_recurrent_quantile_tensor_rollout(native):
    """Native module parity on recurrent multivariate inputs, not a new forecast API.

    The tiny test driver selects the last two context tokens per variate and
    feeds nine trajectories back; subsequent blocks collapse 9x9 samples with
    upstream's torch.quantile rule. Both independent modules use the same test
    driver. This does not repair or substitute for the broken official wrapper.
    """
    torch, _, _, _ = native
    reference = tiny_model(native)
    candidate = copy.deepcopy(reference)
    noisy = copy.deepcopy(reference)
    target = torch.randn(2, 27, 4)
    observed = torch.ones_like(target, dtype=torch.bool)
    samples = torch.ones(2, 27, dtype=torch.long)
    variates = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [0] * 6 + [1] * 6 + [2] * 6).expand(2, -1)
    times = torch.tensor(list(range(3)) * 3 + list(range(3, 9)) * 3).expand(2, -1)
    prediction = torch.zeros(2, 27, dtype=torch.bool)
    prediction[:, 9:] = True
    target[:, 9:] = 0
    # Different variates have ragged, whole-patch observed histories.
    target[0, [0, 3]] = 0
    observed[0, [0, 3]] = False
    samples[0, [0, 3]] = 0

    def rollout(model):
        x, obs, sample, time, var, pred = [item.clone() for item in
                                          (target, observed, samples, times, variates, prediction)]
        anchors = torch.tensor([2, 5, 8])
        blocks = []
        for block in range(3):
            output = model(x, obs, sample, time, var, pred, training_mode=False)
            # Native output: [*batch, sequence, predict_token*quantile*patch].
            values = output.reshape(*output.shape[:-1], 2, 9, 4)[..., anchors, :, :, :]
            values = values.flatten(-4, -3)  # [B,(Qprev), V*2,9,4]
            if block:
                # [B,9prev,V*2,9,4] -> [B,V*2,81,4] -> [9,B,V*2,4]
                values = torch.quantile(values.permute(0, 2, 1, 3, 4).flatten(2, 3),
                                        torch.tensor(model.quantile_levels), dim=-2).permute(1, 2, 0, 3)
            blocks.append(values.clone())
            if block == 0:
                x, obs, sample, time, var, pred = [item.unsqueeze(1).expand(-1, 9, *item.shape[1:]).clone()
                                                 for item in (x, obs, sample, time, var, pred)]
            positions = torch.tensor([9 + block * 2, 10 + block * 2,
                                      15 + block * 2, 16 + block * 2,
                                      21 + block * 2, 22 + block * 2])
            x[..., positions, :] = values.permute(0, 2, 1, 3)
            pred[..., positions] = False
            anchors = positions.reshape(3, 2)[:, -1]
        return blocks

    def one_ulp(module, args, output):
        return torch.nextafter(output, torch.full_like(output, float('inf')))

    hooks = [layer.self_attn.register_forward_hook(one_ulp) for layer in noisy.encoder.layers]
    with torch.inference_mode():
        expected, float_noise = rollout(reference), rollout(noisy)
        undo = install(candidate)
        actual = rollout(candidate)
    assert any((a - b).abs().max() > 0 for a, b in zip(expected, float_noise))
    for expected_block, actual_block, noise_block in zip(expected, actual, float_noise):
        torch.testing.assert_close(actual_block, expected_block, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(noise_block, expected_block, rtol=RTOL, atol=ATOL)
    for hook in hooks:
        hook.remove()
    undo()


def test_inference_and_fp32_guards(native):
    torch, _, _, packed_mask = native
    model = tiny_model(native)
    install(model)
    attention = model.encoder.layers[0].self_attn
    hidden = torch.randn(1, 3, 64)
    ids = torch.zeros(1, 3, dtype=torch.long)
    kwargs = dict(attn_mask=packed_mask(ids, ids), query_var_id=ids, kv_var_id=ids)
    with pytest.raises(ValueError, match='inference-only'):
        attention(hidden, hidden, hidden, **kwargs)
    with torch.inference_mode(), pytest.raises(ValueError, match='FP32 inputs'):
        attention(hidden.half(), hidden.half(), hidden.half(), **kwargs)
    with torch.inference_mode(), torch.autocast('cpu'), pytest.raises(ValueError, match='autocast disabled'):
        attention(hidden, hidden, hidden, **kwargs)
    model.train()
    with torch.inference_mode(), pytest.raises(ValueError, match='eval mode'):
        attention(hidden, hidden, hidden, **kwargs)
