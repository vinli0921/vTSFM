"""CPU regressions against the pinned official TimesFM forward algorithms.

These use small random official modules, not checkpoints or GPU performance
measurements. Patch/head geometry matches the real forecast-block stepper.
"""

import dataclasses
import gc
import weakref

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


def _heterogeneous_prefills(model, capacities=(12, 14, 10), positions=(3, 7, 1)):
    generator = torch.Generator().manual_seed(4949)
    states = []
    for position, masked, capacity in zip(positions, (1, 2, 0), capacities, strict=True):
        caches = _new_caches(model, 1, capacity)
        inputs = torch.randn(1, position, model.p, generator=generator)
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        masks[:, :masked] = True
        model(inputs, masks, caches)
        for cache in caches:
            torch.testing.assert_close(cache.next_index, caches[0].next_index, rtol=0, atol=0)
            torch.testing.assert_close(cache.num_masked, caches[0].num_masked, rtol=0, atol=0)
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
def test_grouped_positions_match_independent_native_rows_and_expose_old_bug(model):
    validate_block_decode_source(model)
    groups, inputs, masks = _heterogeneous_prefills(model)
    before = [_clone_caches(caches) for caches in groups]
    expected, serial_caches = _serial_forward(model, inputs, masks, groups)
    pointers = [[(cache.key.data_ptr(), cache.value.data_ptr()) for cache in caches]
                for caches in groups]
    actual, returned = block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
    assert returned is groups
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    for row, (caches, position) in enumerate(zip(returned, [3, 7, 1], strict=True)):
        for layer, cache in enumerate(caches):
            assert cache is groups[row][layer]
            assert (cache.key.data_ptr(), cache.value.data_ptr()) == pointers[row][layer]
            assert cache.next_index.dtype == torch.int32
            assert cache.num_masked.dtype == torch.int64
            torch.testing.assert_close(cache.next_index, torch.tensor([position + 4], dtype=torch.int32))
            torch.testing.assert_close(cache.num_masked, serial_caches[row][layer].num_masked)
            for name in ('key', 'value'):
                tensor = getattr(cache, name)
                original = getattr(before[row][layer], name)
                reference = getattr(serial_caches[row][layer], name)
                torch.testing.assert_close(tensor[:, :position], original[:, :position], rtol=0, atol=0)
                torch.testing.assert_close(tensor[:, position:position + 4], reference[:, position:position + 4], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(tensor[:, position + 4:], original[:, position + 4:], rtol=0, atol=0)

    # Packing is used only to reproduce the broken upstream negative control.
    wrong, wrong_caches = model(inputs, masks, _merge_caches(before))
    assert torch.count_nonzero(wrong_caches[0].key[1, 7:11]) == 0
    assert torch.count_nonzero(returned[1][0].key[0, 7:11]) > 0
    assert (wrong[2][1] - expected[2][1]).abs().max().item() > 1e-4


@torch.inference_mode()
def test_uniform_group_positions_match_independent_native_requests(model):
    groups, inputs, masks = _heterogeneous_prefills(model, positions=(3, 3, 3))
    expected, _ = _serial_forward(model, inputs, masks, groups)
    actual, _ = block_decode_forward(model, inputs, masks, groups, [3, 3, 3])
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)


@torch.inference_mode()
@pytest.mark.parametrize('capacities', [(18, 22, 18), (22, 22, 22)], ids=['separate', 'packed'])
def test_reordered_groups_and_repeated_advances_preserve_positions_and_masks(model, capacities):
    groups, inputs, masks = _heterogeneous_prefills(model, capacities)
    masks[1, 0] = True  # Preserve native cumulative masked-count promotion.
    expected, serial_caches = _serial_forward(model, inputs, masks, groups)
    order = torch.tensor([2, 0, 1])
    reordered = [groups[i] for i in (2, 0, 1)]
    actual, actual_caches = block_decode_forward(model, inputs[order], masks[order], reordered, [1, 3, 7])
    inverse = torch.argsort(order)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value[inverse], reference, rtol=1e-5, atol=1e-6)
    second_inputs, second_masks = -inputs, torch.zeros_like(masks)
    expected_second, serial_second = _serial_forward(model, second_inputs, second_masks, serial_caches)
    actual_second, second_caches = block_decode_forward(
        model, second_inputs[order], second_masks[order], actual_caches, [5, 7, 11],
    )
    for value, reference in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(value[inverse], reference, rtol=1e-5, atol=1e-6)
    for row, position in enumerate([11, 15, 9]):
        for layer, cache in enumerate(groups[row]):
            torch.testing.assert_close(cache.next_index, torch.tensor([position], dtype=torch.int32))
            torch.testing.assert_close(cache.num_masked, serial_second[row][layer].num_masked)
    assert second_caches is reordered


@torch.inference_mode()
def test_neural_layers_batch_once_and_native_masks_are_reused_across_layers(model, monkeypatch):
    from vtsfm.adapters.timesfm25 import block_decode

    groups, inputs, masks = _heterogeneous_prefills(model)
    calls, projection_calls, hooks, created_masks = [], {}, [], []
    original_mask = block_decode.make_attn_mask

    def record_mask(**kwargs):
        mask = original_mask(**kwargs)
        created_masks.append(mask)
        return mask

    monkeypatch.setattr(block_decode, 'make_attn_mask', record_mask)
    projections = [('tokenizer', model.tokenizer), ('point', model.output_projection_point),
                   ('spread', model.output_projection_quantiles)]
    for index, layer in enumerate(model.stacked_xf):
        names = ('qkv_proj',) if layer.attn.fuse_qkv else ('query', 'key', 'value')
        projections.extend((f'{index}.{name}', getattr(layer.attn, name)) for name in names)
        projections.extend((f'{index}.{name}', getattr(layer, name)) for name in ('ff0', 'ff1'))
        projections.append((f'{index}.out', layer.attn.out))
        original = layer.attn.attention_fn

        def record(query, key, value, mask=None, original=original, index=index):
            calls.append((index, query.shape, key.data_ptr(), value.data_ptr(), key.shape,
                          mask.data_ptr(), mask.clone()))
            return original(query, key, value, mask=mask)

        layer.attn.attention_fn = record
    for name, module in projections:
        projection_calls[name] = []

        def record_projection(module, args, output, name=name):
            projection_calls[name].append(args[0].shape[0])

        hooks.append(module.register_forward_hook(record_projection))
    try:
        block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
    finally:
        for hook in hooks:
            hook.remove()
    assert all(rows == [3] for rows in projection_calls.values())
    assert len(calls) == model.x * len(groups)
    assert len(created_masks) == len(groups)  # One native mask per request, not per layer.
    for index, query_shape, key_ptr, value_ptr, key_shape, mask_ptr, mask in calls:
        row = next(row for row, caches in enumerate(groups) if caches[index].key.data_ptr() == key_ptr)
        cache = groups[row][index]
        assert value_ptr == cache.value.data_ptr()
        assert mask_ptr == created_masks[row].data_ptr()
        assert key_shape == cache.key.shape
        assert query_shape == (1, 4, model.h, model.hd)
        position, masked = [(3, 1), (7, 2), (1, 0)][row]
        expected_mask = torch.zeros(1, 1, 4, cache.key.shape[1], dtype=torch.bool)
        for query in range(4):
            expected_mask[:, :, query, masked:position + query + 1] = True
        torch.testing.assert_close(mask, expected_mask)
    assert [call[4][1] for call in calls[:3]] == [12, 14, 10]


@torch.inference_mode()
def test_equal_width_native_attention_uses_temporary_packs_and_one_mask(model, monkeypatch):
    from vtsfm.adapters.timesfm25 import block_decode

    groups, inputs, masks = _heterogeneous_prefills(model, (18, 18, 18))
    masks[1, 0] = True
    expected, serial_caches = _serial_forward(model, inputs, masks, groups)
    originals = [_clone_caches(caches) for caches in groups]
    pointers = [[(cache.key.data_ptr(), cache.value.data_ptr()) for cache in caches]
                for caches in groups]
    original_inputs, original_masks = inputs.clone(), masks.clone()
    calls, packed_refs, created_masks = [], [], []
    original_mask = block_decode.make_attn_mask

    def record_mask(**kwargs):
        mask = original_mask(**kwargs)
        created_masks.append(mask.clone())
        return mask

    monkeypatch.setattr(block_decode, 'make_attn_mask', record_mask)
    for index, layer in enumerate(model.stacked_xf):
        original = layer.attn.attention_fn

        def record(query, key, value, mask=None, original=original, index=index):
            assert query.shape == (3, 4, model.h, model.hd)
            assert key.shape == value.shape == (3, 18, model.h, model.hd)
            assert key.data_ptr() not in [group[index].key.data_ptr() for group in groups]
            assert value.data_ptr() not in [group[index].value.data_ptr() for group in groups]
            torch.testing.assert_close(key, torch.cat([group[index].key for group in groups]), rtol=0, atol=0)
            torch.testing.assert_close(value, torch.cat([group[index].value for group in groups]), rtol=0, atol=0)
            calls.append((index, mask.data_ptr(), mask.clone()))
            packed_refs.extend(weakref.ref(tensor) for tensor in (key, value, mask))
            return original(query, key, value, mask=mask)

        monkeypatch.setattr(layer.attn, 'attention_fn', record)
    actual, returned = block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
    assert returned is groups
    assert len(calls) == model.x
    assert len(created_masks) == len(groups)
    assert len({mask_ptr for _, mask_ptr, _ in calls}) == 1
    for _, _, mask in calls:
        torch.testing.assert_close(mask, torch.cat(created_masks), rtol=0, atol=0)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    for row, (caches, position) in enumerate(zip(groups, [3, 7, 1], strict=True)):
        for layer, cache in enumerate(caches):
            assert (cache.key.data_ptr(), cache.value.data_ptr()) == pointers[row][layer]
            for name in ('key', 'value'):
                tensor, saved = getattr(cache, name), getattr(originals[row][layer], name)
                reference = getattr(serial_caches[row][layer], name)
                torch.testing.assert_close(tensor[:, :position], saved[:, :position], rtol=0, atol=0)
                torch.testing.assert_close(tensor[:, position:position + 4], reference[:, position:position + 4], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(tensor[:, position + 4:], saved[:, position + 4:], rtol=0, atol=0)
            torch.testing.assert_close(cache.next_index, serial_caches[row][layer].next_index, rtol=0, atol=0)
            torch.testing.assert_close(cache.num_masked, serial_caches[row][layer].num_masked, rtol=0, atol=0)
    torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
    torch.testing.assert_close(masks, original_masks, rtol=0, atol=0)
    gc.collect()
    assert all(reference() is None for reference in packed_refs)


@torch.inference_mode()
def test_hoisted_metadata_is_rebuilt_for_changed_calls_and_not_retained(model, monkeypatch):
    from vtsfm.adapters.timesfm25 import block_decode

    original = block_decode._decode_metadata
    refs = []

    def observe(*args):
        result = original(*args)
        refs.extend(weakref.ref(tensor) for tensor in (result[0], result[1], *result[2]))
        return result

    monkeypatch.setattr(block_decode, '_decode_metadata', observe)
    groups, inputs, masks = _heterogeneous_prefills(model, (18, 22, 18))
    reference, serial = _serial_forward(model, inputs, masks, groups)
    actual, _ = block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
    for value, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
    gc.collect()
    assert len(refs) == 5 and all(ref() is None for ref in refs)
    changed_inputs, changed_masks = -inputs, masks.clone()
    changed_masks[1, 0] = True
    reference, _ = _serial_forward(model, changed_inputs, changed_masks, serial)
    actual, _ = block_decode_forward(model, changed_inputs, changed_masks, groups, [7, 11, 5])
    for value, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
    gc.collect()
    assert len(refs) == 10 and all(ref() is None for ref in refs)


@torch.inference_mode()
@pytest.mark.parametrize('capacities', [(12, 14, 10), (14, 14, 14)], ids=['separate', 'packed'])
def test_grouped_helper_never_extracts_tensor_counter_to_host(model, capacities):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoHostScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError('decode read a Tensor scalar back to the host')
            return func(*args, **(kwargs or {}))

    groups, inputs, masks = _heterogeneous_prefills(model, capacities)
    with NoHostScalar():
        block_decode_forward(model, inputs, masks, groups, [3, 7, 1])


@torch.inference_mode()
def test_decode_counters_advance_once_per_owner_without_layerwise_writes(model):
    from torch.utils._python_dispatch import TorchDispatchMode

    groups, inputs, masks = _heterogeneous_prefills(model, (18, 18, 18))
    counter_ptrs = {cache.next_index.data_ptr() for group in groups for cache in group}
    writes = []

    class CounterWrites(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.add_.Tensor and args[0].data_ptr() in counter_ptrs:
                writes.append(args[0].data_ptr())
            return func(*args, **(kwargs or {}))

    expected, serial = _serial_forward(model, inputs, masks, groups)
    with CounterWrites():
        actual, returned = block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
    assert writes == []  # Counter advancement does not dispatch for every layer.
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    for owner, reference in zip(returned, serial, strict=True):
        for cache, saved in zip(owner, reference, strict=True):
            torch.testing.assert_close(cache.next_index, saved.next_index, rtol=0, atol=0)
            torch.testing.assert_close(cache.num_masked, saved.num_masked, rtol=0, atol=0)


@torch.inference_mode()
def test_returned_decode_counters_can_continue_through_native_forward(model):
    groups, inputs, masks = _heterogeneous_prefills(model, (22, 22, 22))
    block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
    # A graph/experimental/native fallback may update each layer in place.
    # Cross-layer aliases would increment later layers' counters prematurely.
    followup = -inputs[:1]
    expected, expected_caches = model(followup, masks[:1], _clone_caches(groups[0]))
    actual, actual_caches = model(followup, masks[:1], groups[0])
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    for cache, reference in zip(actual_caches, expected_caches, strict=True):
        torch.testing.assert_close(cache.next_index, reference.next_index, rtol=0, atol=0)
        torch.testing.assert_close(cache.num_masked, reference.num_masked, rtol=0, atol=0)


@torch.inference_mode()
def test_native_model_parameters_methods_and_whole_decode_unchanged(model):
    source_methods = (type(model).forward, type(model.stacked_xf[0]).forward,
                      type(model.stacked_xf[0].attn).forward)
    instance_methods = [module.__dict__.get('forward') for module in model.modules()]
    parameters = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    inputs = torch.linspace(-1, 1, 64).reshape(1, -1)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    before = model.decode(129, inputs, masks)
    groups, patched, patch_masks = _heterogeneous_prefills(model)
    block_decode_forward(model, patched, patch_masks, groups, [3, 7, 1])
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
    groups, inputs, masks = _heterogeneous_prefills(model)
    expected, _ = _serial_forward(model, inputs, masks, groups)

    def wrapped(*args, **kwargs):
        raise AssertionError('instance forward wrapper should not be inspected or replaced')

    model.forward = wrapped
    model.stacked_xf[0].forward = wrapped
    model.stacked_xf[0].attn.forward = wrapped
    validate_block_decode_source(model)
    actual, _ = block_decode_forward(model, inputs, masks, groups, [3, 7, 1])
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


@pytest.mark.parametrize('invalid', ['capacity', 'layer_capacity', 'rows', 'count', 'position'])
@torch.inference_mode()
def test_invalid_group_geometry_fails_before_mutating_any_cache(model, invalid):
    groups, inputs, masks = _heterogeneous_prefills(model)
    before = [_clone_caches(caches) for caches in groups]
    positions = [3, 7, 1]
    if invalid == 'capacity':
        positions[-1] = 7
    elif invalid == 'layer_capacity':
        groups[-1][-1].key = groups[-1][-1].key[:, :-1]
        groups[-1][-1].value = groups[-1][-1].value[:, :-1]
        before[-1] = _clone_caches(groups[-1])
    elif invalid == 'rows':
        inputs, masks = inputs[:2], masks[:2]
    elif invalid == 'count':
        positions.pop()
    else:
        positions[0] = torch.tensor(3, dtype=torch.int32)
    with pytest.raises(ValueError, match='cache|Cache'):
        block_decode_forward(model, inputs, masks, groups, positions)
    for row, caches in enumerate(groups):
        original = before[row]
        for cache, saved in zip(caches, original, strict=True):
            for name in ('key', 'value', 'next_index', 'num_masked'):
                torch.testing.assert_close(getattr(cache, name), getattr(saved, name), rtol=0, atol=0)
