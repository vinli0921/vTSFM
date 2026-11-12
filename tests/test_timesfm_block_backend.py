"""CPU coverage for the shared neural executor's optional attention boundary."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timesfm')

from timesfm.torch.transformer import make_attn_mask  # noqa: E402
from timesfm.torch.util import DecodeCache  # noqa: E402

from test_timesfm_block_decode import (  # noqa: E402
    _clone_caches,
    _heterogeneous_prefills,
    tiny_model,
)
from vtsfm.adapters.timesfm25.block_decode import (  # noqa: E402
    _decode_positions,
    block_decode_forward,
)


@pytest.mark.parametrize('fused_qkv', [True, False])
@torch.inference_mode()
def test_optional_attention_keeps_neural_batching_and_counter_semantics(fused_qkv):
    """Empty dense caches still share QKV/FFN/heads and native integer metadata.

    The test boundary owns ordinary separate payload tensors to provide a CPU
    reference. Real page allocation and Triton attention are qualified manually.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny_model(fused_qkv)
        native_groups, inputs, masks = _heterogeneous_prefills(model, (18, 18, 18))
        payloads = [_clone_caches(caches) for caches in native_groups]
        empty_groups = [[DecodeCache(
            cache.next_index.clone(), cache.num_masked.clone(),
            cache.key[:, :0].clone(), cache.value[:, :0].clone(),
        ) for cache in caches] for caches in native_groups]
        positions = [3, 7, 1]
        masks[1, 0] = True
        expected, expected_caches = block_decode_forward(model, inputs, masks, native_groups, positions)

        class Boundary:
            prepared = 0
            attended = []

            def prepare(self, selected_model, batch_inputs, batch_masks, caches, starts):
                assert selected_model is model
                assert caches is empty_groups and starts == positions
                self.prepared += 1
                groups = [(slice(row, row + 1), group, start)
                          for row, (group, start) in enumerate(zip(caches, starts, strict=True))]
                position, num_masked, next_index = _decode_positions(batch_inputs, batch_masks, groups)
                attention_masks = [make_attn_mask(
                    query_length=model.m, num_all_masked_kv=num_masked[section],
                    query_index_offset=next_index[section], kv_length=payloads[row][0].key.shape[1],
                ) for row, (section, _, _) in enumerate(groups)]
                return groups, position, num_masked, torch.cat(attention_masks)

            def attend(self, layer, query, key, value, state):
                assert query.shape == key.shape == value.shape == (3, model.m, model.h, model.hd)
                self.attended.append(layer)
                for row, start in enumerate(positions):
                    payloads[row][layer].key[:, start:start + model.m].copy_(key[row:row + 1])
                    payloads[row][layer].value[:, start:start + model.m].copy_(value[row:row + 1])
                return model.stacked_xf[layer].attn.attention_fn(
                    query, torch.cat([group[layer].key for group in payloads]),
                    torch.cat([group[layer].value for group in payloads]), mask=state)

        boundary = Boundary()
        projections = [model.tokenizer, model.output_projection_point, model.output_projection_quantiles]
        for layer in model.stacked_xf:
            projections.extend([layer.ff0, layer.ff1, layer.attn.out])
            projections.extend([layer.attn.qkv_proj] if fused_qkv
                               else [layer.attn.query, layer.attn.key, layer.attn.value])
        calls = [[] for _ in projections]
        hooks = [module.register_forward_hook(
            lambda module, args, output, rows=rows: rows.append(args[0].shape[0]))
            for module, rows in zip(projections, calls, strict=True)]
        try:
            actual, returned = block_decode_forward(
                model, inputs, masks, empty_groups, positions, attention_backend=boundary)
        finally:
            for hook in hooks:
                hook.remove()
        assert boundary.prepared == 1 and boundary.attended == list(range(model.x))
        assert all(rows == [3] for rows in calls)
        assert returned is empty_groups
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
        for row, group in enumerate(empty_groups):
            for layer, cache in enumerate(group):
                assert cache.key.numel() == cache.value.numel() == 0
                torch.testing.assert_close(cache.next_index, expected_caches[row][layer].next_index,
                                           rtol=0, atol=0)
                torch.testing.assert_close(cache.num_masked, expected_caches[row][layer].num_masked,
                                           rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
