"""Experimental direct-backend proof of vLLM-owned TimesFM KV block leases.

This is not a GPUModelRunner or HTTP integration. The arena uses the pinned
vLLM block manager and physical allocator, then calls its Triton attention
backend directly. A logical token is one TimesFM patch, not one forecast block.
All calls and releases must use one CUDA stream; prefix reuse is disabled.
"""

from dataclasses import dataclass, field
from importlib.metadata import version
import os

import torch


@dataclass
class _Lease:
    request: object
    capacity: int
    positions: dict = field(default_factory=dict)


class PagedCacheArena:
    """Independent FP32 caches for each configured stream and decoder layer.

    ``attend`` accepts packed CUDA tensors [sum(query_lengths), heads, head_size].
    ``starts`` are already-cached patch counts, separately checked for each
    sequence/layer/stream. Scale is one because TimesFM scales Q upstream.
    """

    def __init__(self, layers, heads, head_size, max_sequences, max_tokens,
                 block_size=16, device='cuda', dtype=torch.float32, streams=2):
        if version('vllm') != '0.33.0' or torch.__version__.split('+')[0] != '2.14.0':
            raise ValueError('Paged TimesFM proof requires vLLM 0.33.0 and Torch 2.14.0')
        if torch.version.cuda != '13.0':
            raise ValueError('Paged TimesFM proof requires the pinned CUDA 13.0 Torch build')
        if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
            raise ValueError('Launch with TRITON_F32_DEFAULT=ieee before importing Triton')
        if any(type(n) is not int or n < 1
               for n in (layers, heads, head_size, max_sequences, max_tokens, block_size, streams)):
            raise ValueError('Arena dimensions must be positive integers')
        if dtype != torch.float32 or torch.device(device).type != 'cuda':
            raise ValueError('Paged TimesFM proof requires CUDA FP32')

        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
        from vllm.v1.kv_cache_layout import KVCacheLayout
        from vllm.v1.worker.utils import allocate_kv_cache

        if (not TritonAttentionBackend.supports_head_size(head_size)
                or not TritonAttentionBackend.supports_block_size(block_size)):
            raise ValueError('Unsupported Triton attention head or block size')
        self.layers, self.heads, self.head_size = layers, heads, head_size
        self.streams = streams
        self.max_sequences, self.max_tokens = max_sequences, max_tokens
        self.block_size = block_size
        self.device = torch.device('cuda', torch.cuda.current_device()) if str(device) == 'cuda' else torch.device(device)
        self._stream = torch.cuda.current_stream(self.device)
        self._leases = {}
        self._plan_key = self._plan_metadata = None
        self._closed = False
        self._names = {(layer, stream): f'proof.{stream}.layers.{layer}.attn'
                       for stream in range(streams) for layer in range(layers)}
        spec = FullAttentionSpec(block_size=block_size, num_kv_heads=heads,
                                 head_size=head_size, dtype=dtype)
        # Block zero is the manager's null block. Every real sequence can reserve
        # its complete rounded capacity without borrowing a peer's pages.
        count = max_sequences * ((max_tokens + block_size - 1) // block_size) + 1
        self._page_bytes = spec.page_size_bytes
        names = list(self._names.values())
        config = KVCacheConfig(
            num_blocks=count,
            kv_cache_tensors=[KVCacheTensor(
                size=count * spec.page_size_bytes * len(names), layers=names,
                layer_stride=count * spec.page_size_bytes,
                block_stride=spec.page_size_bytes)],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
        )
        self.manager = KVCacheManager(
            config, max_model_len=max_tokens, scheduler_block_size=block_size,
            hash_block_size=block_size, enable_caching=False)
        # vLLM 0.33 has no backend.get_kv_cache_shape: its official allocator
        # returns logical [blocks, heads, block_size, 2*head_size] views instead.
        self.caches = allocate_kv_cache(config, self.device, KVCacheLayout.LBHNC)
        self.impl = TritonAttentionImpl(
            num_heads=heads, head_size=head_size, scale=1.0,
            num_kv_heads=heads, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype='auto')
        # Direct backend calls need the layer's identity quantization scales.
        self._layer = torch.nn.Module()
        for name in ('_q_scale', '_k_scale', '_v_scale'):
            self._layer.register_buffer(name, torch.ones((), dtype=dtype, device=self.device))

    @property
    def allocated(self):
        return {key: self.block_ids(key) for key in self._leases}

    def block_ids(self, sequence_id):
        return tuple(self.manager.get_block_ids(sequence_id)[0])

    @property
    def free_block_count(self):
        return self.manager.block_pool.get_num_free_blocks()

    @property
    def cache_bytes(self):
        """Fixed physical pool bytes, retained even when all leases are free."""
        return sum(cache.numel() * cache.element_size() for cache in self.caches.values())

    @property
    def leased_cache_bytes(self):
        return sum(len(ids) for ids in self.allocated.values()) * self._page_bytes * self.streams * self.layers

    def reserve(self, sequence_id, capacity):
        from vllm import PoolingParams
        from vllm.v1.request import Request

        if self._closed:
            raise ValueError('Arena is closed')
        if not isinstance(sequence_id, str) or not sequence_id or sequence_id in self._leases:
            raise ValueError('Reserve requires a new nonempty sequence ID')
        if type(capacity) is not int or not 1 <= capacity <= self.max_tokens:
            raise ValueError('Sequence capacity exceeds the arena token limit')
        if len(self._leases) >= self.max_sequences:
            raise ValueError('Arena sequence limit exceeded')
        request = Request(sequence_id, [0] * capacity, sampling_params=None,
                          pooling_params=PoolingParams(task='plugin'))
        if self.manager.allocate_slots(request, capacity) is None:
            raise RuntimeError('vLLM KV block pool cannot satisfy the reservation')
        self._leases[sequence_id] = _Lease(request, capacity)
        self._plan_key = self._plan_metadata = None
        return self.block_ids(sequence_id)

    def _check_stream(self):
        if torch.cuda.current_stream(self.device) != self._stream:
            raise ValueError('Arena cache writes and release must use the original CUDA stream')

    def release(self, sequence_id):
        self._check_stream()
        self._plan_key = self._plan_metadata = None
        lease = self._leases.pop(sequence_id, None)
        if lease is not None:
            self.manager.free(lease.request)

    def close(self):
        """Return all leases and drop the owned physical pool between probes."""
        if self._closed:
            return
        self._check_stream()
        for sequence_id in tuple(self._leases):
            self.release(sequence_id)
        self._plan_key = self._plan_metadata = None
        self.caches.clear()
        self._layer = None
        self._closed = True

    def _metadata(self, sequence_ids, starts, query_lengths, cumulative, lengths):
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

        plan_key = (sequence_ids, starts, query_lengths)
        if plan_key == self._plan_key:
            return self._plan_metadata
        block_ids, slots = [], []
        for sequence_id, start, length in zip(sequence_ids, starts, query_lengths):
            ids = self.block_ids(sequence_id)
            block_ids.append(ids)
            slots.extend(ids[position // self.block_size] * self.block_size + position % self.block_size
                         for position in range(start, start + length))
        width = max(map(len, block_ids))
        table = torch.tensor([list(ids) + [0] * (width - len(ids)) for ids in block_ids],
                             dtype=torch.int32, device=self.device)
        mapping = torch.tensor(slots, dtype=torch.int64, device=self.device)
        # All layer/flip views share block IDs, so one cohort plan serves every
        # attention layer. Retain only this plan; lease changes invalidate it.
        # None scratch selects 2D attention; tuned split-K is a separate probe.
        metadata = TritonAttentionMetadata(
            num_actual_tokens=cumulative[-1], max_query_len=max(query_lengths),
            query_start_loc=torch.tensor(cumulative, dtype=torch.int32, device=self.device),
            max_seq_len=max(lengths), seq_lens=torch.tensor(lengths, dtype=torch.int32, device=self.device),
            block_table=table, slot_mapping=mapping, causal=True,
            seq_threshold_3D=None, num_par_softmax_segments=None,
            softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None)
        self._plan_key, self._plan_metadata = plan_key, metadata
        return metadata

    @torch.inference_mode()
    def attend(self, layer, stream, query, key, value, sequence_ids, starts, query_lengths):
        self._check_stream()
        if self._closed:
            raise ValueError('Arena is closed')
        if (layer, stream) not in self._names:
            raise ValueError('Invalid decoder layer or flip stream')
        sequence_ids, starts, query_lengths = tuple(sequence_ids), tuple(starts), tuple(query_lengths)
        if (not sequence_ids or len(set(sequence_ids)) != len(sequence_ids)
                or len(sequence_ids) != len(starts) or len(starts) != len(query_lengths)):
            raise ValueError('Attention requires unique sequence IDs and matching metadata lengths')
        lengths, cumulative = [], [0]
        for sequence_id, start, length in zip(sequence_ids, starts, query_lengths):
            lease = self._leases.get(sequence_id)
            if (lease is None or type(start) is not int or type(length) is not int
                    or start < 0 or length < 1 or start + length > lease.capacity):
                raise ValueError('Attention exceeds a reserved sequence capacity')
            if start != lease.positions.get((layer, stream), 0):
                raise ValueError('Attention must append to the valid sequence cache prefix')
            lengths.append(start + length)
            cumulative.append(cumulative[-1] + length)
        shape = (cumulative[-1], self.heads, self.head_size)
        if any(not isinstance(t, torch.Tensor) or tuple(t.shape) != shape
               or t.dtype != torch.float32 or t.device != self.device for t in (query, key, value)):
            raise ValueError('Attention Q/K/V must be packed CUDA FP32 tensors with the arena head shape')
        metadata = self._metadata(sequence_ids, starts, query_lengths, cumulative, lengths)
        cache = self.caches[self._names[layer, stream]]
        self.impl.do_kv_cache_update(self._layer, key, value, cache, metadata.slot_mapping)
        output = torch.empty_like(query)
        self.impl.forward(self._layer, query, key, value, cache, metadata, output=output)
        for sequence_id, length in zip(sequence_ids, lengths):
            self._leases[sequence_id].positions[layer, stream] = length
        return output
