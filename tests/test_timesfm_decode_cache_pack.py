"""Copy-only CUDA packing must preserve each independent owner's cache bytes."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from vtsfm.adapters.timesfm25.decode_cache import prepare_decode_cache_pack  # noqa: E402


def caches(rows, capacity, *, layers=2, device='cpu'):
    return [SimpleNamespace(
        key=torch.arange(rows * capacity * 16, device=device, dtype=torch.float32).reshape(rows, capacity, 2, 8)
        + 1000 * layer,
        value=-torch.arange(rows * capacity * 16, device=device, dtype=torch.float32).reshape(rows, capacity, 2, 8)
        - 1000 * layer,
    ) for layer in range(layers)]


def test_cpu_decode_does_not_require_triton():
    groups = [(slice(0, 2), caches(2, 12), 3)]
    assert prepare_decode_cache_pack(groups, torch.empty(2, 4, 32)) is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@torch.inference_mode()
def test_append_pack_preserves_mixed_owner_bytes_and_reuses_only_call_scratch():
    # The first owner has a strided row view, as with the singleton flip halves.
    first, second = caches(4, 19, device='cuda'), caches(1, 19, device='cuda')
    for cache in first:
        cache.key, cache.value = cache.key[::2], cache.value[::2]
    groups = [(slice(0, 2), first, 3), (slice(2, 3), second, 11)]
    inputs = torch.empty(3, 4, 32, device='cuda')
    plan = prepare_decode_cache_pack(groups, inputs)
    assert plan is not None
    original = [[(cache.key.clone(), cache.value.clone()) for cache in owner] for owner in (first, second)]
    pointers = [[(cache.key.data_ptr(), cache.value.data_ptr()) for cache in owner] for owner in (first, second)]
    scratch = None
    for layer in range(2):
        # QKV projection chunks are not contiguous along the patch axis.
        qkv = torch.randn(3, 4, 48, device='cuda')
        key, value = [tensor.reshape(3, 4, 2, 8) for tensor in qkv.chunk(3, dim=-1)[1:]]
        packed = plan.append_and_pack(layer, key, value)
        addresses = tuple(tensor.data_ptr() for tensor in packed)
        if scratch is not None:
            assert addresses == scratch
        scratch = addresses
        expected_keys, expected_values = [], []
        for owner_index, (section, owner, start) in enumerate(groups):
            saved_key, saved_value = original[owner_index][layer]
            saved_key[:, start:start + 4] = key[section]
            saved_value[:, start:start + 4] = value[section]
            torch.testing.assert_close(owner[layer].key, saved_key, rtol=0, atol=0)
            torch.testing.assert_close(owner[layer].value, saved_value, rtol=0, atol=0)
            assert (owner[layer].key.data_ptr(), owner[layer].value.data_ptr()) == pointers[owner_index][layer]
            expected_keys.append(saved_key)
            expected_values.append(saved_value)
        torch.testing.assert_close(packed[0], torch.cat(expected_keys), rtol=0, atol=0)
        torch.testing.assert_close(packed[1], torch.cat(expected_values), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_unequal_capacity_and_graph_capture_keep_reference_path():
    groups = [(slice(0, 1), caches(1, 12, device='cuda'), 3),
              (slice(1, 2), caches(1, 19, device='cuda'), 11)]
    inputs = torch.empty(2, 4, 32, device='cuda')
    assert prepare_decode_cache_pack(groups, inputs) is None
    groups[0] = (slice(0, 1), caches(1, 19, device='cuda'), 3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert prepare_decode_cache_pack(groups, inputs) is None
    torch.cuda.current_stream().wait_stream(stream)
