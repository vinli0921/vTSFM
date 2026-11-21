"""Call-local, copy-only CUDA append/pack for independent dense decode caches.

Attention still uses the native masks and FP32 kernel. Only new K/V patches are
written into owned caches; full-width gathers go into scratch reused across the
layers of one forward. Prefill, CUDA graphs and unequal capacities stay on the
reference path. No pointer table or scratch is retained by a request or model.
"""

import torch


class _DecodeCachePack:
    def __init__(self, groups, inputs):
        sample = groups[0][1][0].key
        self.rows, self.capacity = inputs.shape[0], sample.shape[1]
        self.heads, self.dim = sample.shape[2:]
        pointers = []
        for layer in range(len(groups[0][1])):
            layer_rows = []
            for _, caches, start in groups:
                key, value = caches[layer].key, caches[layer].value
                for row in range(key.shape[0]):
                    layer_rows.append((key.data_ptr() + row * key.stride(0) * 4,
                                       value.data_ptr() + row * value.stride(0) * 4, start))
            pointers.append(layer_rows)
        # This is the sole metadata H2D copy, once per forward rather than layer.
        # Blocking staging keeps the host source alive through its copy. The
        # device table and scratch then live on the caller's executing stream.
        self.pointers = torch.tensor(pointers, dtype=torch.int64, device='cpu').to(inputs.device)
        self.scratch = torch.empty((2, self.rows, self.capacity, self.heads, self.dim),
                                   device=inputs.device, dtype=torch.float32)

    def append_and_pack(self, layer, key, value):
        from .decode_cache_kernel import append_and_pack

        packed_key, packed_value = self.scratch.unbind(0)
        append_and_pack(self.pointers, layer, key, value, packed_key, packed_value)
        return packed_key, packed_value


def prepare_decode_cache_pack(groups, inputs):
    """Return CUDA FP32 packing scratch or ``None`` for the reference path.

    ``groups`` has the validated ``(row_slice, layer_caches, CPU_write_position)``
    structure from block_decode. Metadata is O(layers * rows); scratch is exactly
    two packed K/V buffers, the same payload as the reference's two ``cat`` calls.
    Row-strided views of owned caches are supported; each row must be dense.
    """
    if inputs.device.type != 'cuda' or inputs.dtype != torch.float32:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    first = groups[0][1][0].key
    capacity, heads, dim = first.shape[1:]
    for section, caches, start in groups:
        for cache in caches:
            for tensor in (cache.key, cache.value):
                if (tensor.device != inputs.device or tensor.dtype != torch.float32
                        or tensor.shape != (section.stop - section.start, capacity, heads, dim)
                        or tensor.stride()[1:] != (heads * dim, dim, 1)
                        or tensor.stride(0) < capacity * heads * dim):
                    return None
    return _DecodeCachePack(groups, inputs)
