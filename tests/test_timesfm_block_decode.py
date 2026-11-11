"""CPU regressions against the pinned official TimesFM forward algorithms.

These use small random official modules, not checkpoints or GPU performance
measurements. Patch/head geometry matches the real forecast-block stepper.
"""

import dataclasses

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timesfm')

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module  # noqa: E402
from timesfm.torch.normalization import RMSNorm  # noqa: E402
from timesfm.torch.util import DecodeCache  # noqa: E402

from vtsfm.adapters.timesfm25.block_decode import (  # noqa: E402
    block_decode_forward,
    validate_block_decode_source,
)


def tiny_model(fuse_qkv=True):
    """Official two-layer width-16 model; retain real p/o/os/q/aridx geometry."""
    original = TimesFM_2p5_200M_torch_module.config
    width = 16
    transformer = dataclasses.replace(
        original.stacked_transformers.transformer,
        model_dims=width, hidden_dims=width, num_heads=2, fuse_qkv=fuse_qkv,
    )

    class TinyTimesFM(TimesFM_2p5_200M_torch_module):
        config = dataclasses.replace(
            original,
            tokenizer=dataclasses.replace(original.tokenizer, hidden_dims=width, output_dims=width),
            stacked_transformers=dataclasses.replace(
                original.stacked_transformers, num_layers=2, transformer=transformer,
            ),
            output_projection_point=dataclasses.replace(
                original.output_projection_point, input_dims=width, hidden_dims=width,
            ),
            output_projection_quantiles=dataclasses.replace(
                original.output_projection_quantiles, input_dims=width, hidden_dims=width,
            ),
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(49)
        model = TinyTimesFM().cpu().float().eval()
        # Upstream initializes RMS scales to zero pending checkpoint loading.
        # Nonzero deterministic scales make cache/attention regressions observable.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, RMSNorm):
                    module.scale.copy_(torch.linspace(0.8, 1.2, module.num_features))
        return model


@pytest.fixture(params=[True, False], ids=['fused-qkv', 'separate-qkv'])
def model(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield tiny_model(request.param)
    finally:
        torch.set_num_threads(previous)


def _new_caches(model, rows, capacity):
    return [DecodeCache(
        next_index=torch.zeros(rows, dtype=torch.int32),
        num_masked=torch.zeros(rows, dtype=torch.int32),
        key=torch.zeros(rows, capacity, model.h, model.hd),
        value=torch.zeros(rows, capacity, model.h, model.hd),
    ) for _ in range(model.x)]


def _clone_caches(caches):
    return [DecodeCache(*(value.clone() for value in
                          (cache.next_index, cache.num_masked, cache.key, cache.value)))
            for cache in caches]


def _merge_caches(parts):
    result = []
    capacity = max(caches[0].key.shape[1] for caches in parts)
    for layer in range(len(parts[0])):
        caches = [part[layer] for part in parts]
        keys, values = [], []
        for cache in caches:
            padded_key = torch.zeros(cache.key.shape[0], capacity, *cache.key.shape[2:])
            padded_value = torch.zeros_like(padded_key)
            padded_key[:, :cache.key.shape[1]] = cache.key
            padded_value[:, :cache.value.shape[1]] = cache.value
            keys.append(padded_key)
            values.append(padded_value)
        result.append(DecodeCache(
            torch.cat([cache.next_index for cache in caches]),
            torch.cat([cache.num_masked for cache in caches]),
            torch.cat(keys), torch.cat(values),
        ))
    return result


def _heterogeneous_prefills(model, capacities=(12, 14, 10)):
    generator = torch.Generator().manual_seed(4949)
    states = []
    for (position, masked), capacity in zip(((3, 1), (7, 2), (1, 0)), capacities, strict=True):
        caches = _new_caches(model, 1, capacity)
        inputs = torch.randn(1, position, model.p, generator=generator)
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        masks[:, :masked] = True
        model(inputs, masks, caches)
        states.append(caches)
    inputs = torch.randn(3, model.m, model.p, generator=generator)
    return states, inputs, torch.zeros_like(inputs, dtype=torch.bool)


def _serial_forward(model, inputs, masks, states):
    outputs = []
    caches = []
    for row, state in enumerate(states):
        output, cache = model(inputs[row:row + 1], masks[row:row + 1], _clone_caches(state))
        outputs.append(output)
        caches.append(cache)
    return tuple(torch.cat([output[i] for output in outputs]) for i in range(4)), caches


@torch.inference_mode()
def test_mixed_positions_match_independent_native_rows_and_expose_old_bug(model):
    validate_block_decode_source(model)
    states, inputs, masks = _heterogeneous_prefills(model)
    expected, serial_caches = _serial_forward(model, inputs, masks, states)
    caches = _merge_caches(states)
    original = _clone_caches(caches)
    pointers = [(cache.key.data_ptr(), cache.value.data_ptr()) for cache in caches]
    actual, returned = block_decode_forward(model, inputs, masks, caches)

    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    for layer, cache in enumerate(returned):
        assert cache is caches[layer]
        assert (cache.key.data_ptr(), cache.value.data_ptr()) == pointers[layer]
        assert cache.next_index.dtype == torch.int32
        assert cache.num_masked.dtype == serial_caches[0][layer].num_masked.dtype == torch.int64
        torch.testing.assert_close(cache.next_index, torch.tensor([7, 11, 5], dtype=torch.int32))
        torch.testing.assert_close(cache.num_masked, torch.tensor([1, 2, 0], dtype=torch.int64))
        for row, state in enumerate(states):
            position = int(state[layer].next_index[0])
            for name in ('key', 'value'):
                tensor = getattr(cache, name)
                before = getattr(original[layer], name)
                reference = getattr(serial_caches[row][layer], name)
                torch.testing.assert_close(tensor[row, :position], before[row, :position], rtol=0, atol=0)
                torch.testing.assert_close(tensor[row, position:position + model.m], reference[0, position:position + model.m], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(tensor[row, position + model.m:], before[row, position + model.m:], rtol=0, atol=0)
            # Packed scratch never aliases an independently owned request cache.
            assert cache.key.untyped_storage().data_ptr() != state[layer].key.untyped_storage().data_ptr()

    wrong, wrong_caches = model(inputs, masks, _merge_caches(states))
    # A real regression fixture: upstream writes row1 at row0's index3, leaving
    # row1's actual extension7:11 empty and changing its forecast projection.
    assert torch.count_nonzero(wrong_caches[0].key[1, 7:11]) == 0
    assert torch.count_nonzero(returned[0].key[1, 7:11]) > 0
    assert (wrong[2][1] - expected[2][1]).abs().max().item() > 1e-4


@torch.inference_mode()
def test_prefill_and_uniform_positions_keep_native_results(model):
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn(2, 3, model.p, generator=generator)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    masks[0, 0] = True
    for caches in (None, _new_caches(model, 2, 12)):
        native, native_cache = model(inputs, masks, _clone_caches(caches) if caches else None)
        actual, actual_cache = block_decode_forward(model, inputs, masks, _clone_caches(caches) if caches else None)
        for value, reference in zip(actual, native, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        if caches:
            for a, b in zip(actual_cache, native_cache, strict=True):
                for name in ('key', 'value', 'next_index', 'num_masked'):
                    torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        else:
            assert actual_cache == native_cache == [None] * model.x


@torch.inference_mode()
def test_reordered_rows_and_repeated_advances_preserve_positions_and_masks(model):
    states, inputs, masks = _heterogeneous_prefills(model, (18, 22, 18))
    # Exercise the native cumulative masked-count update as well as zero AR masks.
    masks[1, 0] = True
    expected, serial_caches = _serial_forward(model, inputs, masks, states)
    packed = _merge_caches(states)
    order = torch.tensor([2, 0, 1])
    reordered = [DecodeCache(*(tensor[order].clone() for tensor in
                               (cache.next_index, cache.num_masked, cache.key, cache.value)))
                 for cache in packed]
    actual, actual_caches = block_decode_forward(model, inputs[order], masks[order], reordered)
    inverse = torch.argsort(order)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value[inverse], reference, rtol=1e-5, atol=1e-6)
    for layer, cache in enumerate(actual_caches):
        expected_counts = torch.cat([state[layer].num_masked for state in serial_caches])
        torch.testing.assert_close(cache.num_masked[inverse], expected_counts)
    second_inputs = -inputs
    second_masks = torch.zeros_like(masks)
    expected_second, serial_second = _serial_forward(model, second_inputs, second_masks, serial_caches)
    actual_second, second_caches = block_decode_forward(
        model, second_inputs[order], second_masks[order], actual_caches,
    )
    for value, reference in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(value[inverse], reference, rtol=1e-5, atol=1e-6)
    for layer, cache in enumerate(second_caches):
        torch.testing.assert_close(cache.next_index[inverse], torch.tensor([11, 15, 9], dtype=torch.int32))
        torch.testing.assert_close(cache.num_masked[inverse], torch.cat([state[layer].num_masked for state in serial_second]))


@torch.inference_mode()
def test_native_attention_function_receives_full_capacity_and_row_masks(model):
    states, inputs, masks = _heterogeneous_prefills(model)
    calls = []
    for layer in model.stacked_xf:
        original = layer.attn.attention_fn

        def record(query, key, value, mask=None, original=original):
            calls.append((key.shape, mask.clone()))
            return original(query, key, value, mask=mask)

        layer.attn.attention_fn = record
    block_decode_forward(model, inputs, masks, _merge_caches(states))
    assert len(calls) == model.x
    expected = torch.zeros(3, 1, model.m, 14, dtype=torch.bool)
    for row, (position, num_masked) in enumerate(((3, 1), (7, 2), (1, 0))):
        for query in range(model.m):
            expected[row, 0, query, num_masked:position + query + 1] = True
    for shape, mask in calls:
        assert shape == (3, 14, model.h, model.hd)
        torch.testing.assert_close(mask, expected)


@torch.inference_mode()
def test_native_model_parameters_methods_and_whole_decode_unchanged(model):
    source_methods = (type(model).forward, type(model.stacked_xf[0]).forward,
                      type(model.stacked_xf[0].attn).forward)
    instance_methods = [module.__dict__.get('forward') for module in model.modules()]
    parameters = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    inputs = torch.linspace(-1, 1, 64).reshape(1, -1)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    before = model.decode(129, inputs, masks)
    states, patched, patch_masks = _heterogeneous_prefills(model)
    block_decode_forward(model, patched, patch_masks, _merge_caches(states))
    after = model.decode(129, inputs, masks)
    for value, reference in zip(after, before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, parameters[name], rtol=0, atol=0)
    assert source_methods == (type(model).forward, type(model.stacked_xf[0]).forward,
                              type(model.stacked_xf[0].attn).forward)
    assert instance_methods == [module.__dict__.get('forward') for module in model.modules()]


@torch.inference_mode()
def test_source_guard_reads_class_forward_and_preserves_instance_wrappers(model):
    states, inputs, masks = _heterogeneous_prefills(model)
    expected, _ = _serial_forward(model, inputs, masks, states)
    def wrapped(*args, **kwargs):
        raise AssertionError('instance forward wrapper should not be inspected or replaced')

    model.forward = wrapped
    model.stacked_xf[0].forward = wrapped
    model.stacked_xf[0].attn.forward = wrapped
    validate_block_decode_source(model)
    actual, _ = block_decode_forward(model, inputs, masks, _merge_caches(states))
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    assert model.forward is wrapped
    assert model.stacked_xf[0].forward is wrapped
    assert model.stacked_xf[0].attn.forward is wrapped


def test_source_guard_rejects_changed_class_forward(model):
    class ChangedForward(type(model)):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    with pytest.raises(ValueError, match='pinned TimesFM model.forward'):
        validate_block_decode_source(ChangedForward())
