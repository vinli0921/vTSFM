# Copyright 2026 vTSFM contributors
#
# Attention transformations are derived from Google LLC's TimesFM Torch code
# (Copyright 2025 Google LLC), Apache-2.0. Decoder progression remains in the
# existing TimesFMStepper. See repository LICENSE and google-research/timesfm.
"""Experimental #14 attention/block-allocator proof, not a serving runner.

An owned model instance uses vLLM's Triton backend and KV block manager. The
normalization, transformer layers, heads and forecast assembly stay upstream.
No CLI serving flag selects this implementation. Use only the manual harness.
"""

from types import MethodType

import torch
from timesfm.torch.util import DecodeCache

from .paged_cache import PagedCacheArena
from .stepping import TimesFMStepper


class PagedTimesFMStepper(TimesFMStepper):
    """Keep original forecast state; replace dense KV payloads with page leases.

    Each (series, sign) owns a separate lease in one physical layer view. This
    lets normal/flipped rows share one attention call, just as they share all
    neural layers in the optimized dense control. Signs never share KV bytes.
    Dummy DecodeCaches contain only position counters and empty K/V tensors.
    ``state.cache_bytes`` therefore excludes the arena; report arena.cache_bytes
    and leased_cache_bytes explicitly. All work is on one CUDA stream.
    """

    def __init__(self, runtime, *, max_sequences=2, max_tokens=284):
        if not all(hasattr(TimesFMStepper, method)
                   for method in ('_prefill_passes', '_advance_passes', '_store', '_block_decode_forward')):
            raise ValueError('Native proof requires the recorded optimized stepper and attention seam')
        super().__init__(runtime)
        self.arena = PagedCacheArena(self.model.x, self.model.h, self.model.hd,
                                     2 * max_sequences, max_tokens, streams=1,
                                     device=next(self.model.parameters()).device)
        self._routes = {}
        self._active = None
        self._counter = 0
        self._original = []
        self._closed = False
        for index, layer in enumerate(self.model.stacked_xf):
            attention = layer.attn
            if not (attention.fuse_qkv and not attention.use_bias
                    and attention.use_rotary_position_embeddings and attention.use_per_dim_scale):
                self.close()
                raise ValueError('Native proof requires the pinned fused-QKV TimesFM attention')
            self._original.append((attention, attention.forward))
            attention.forward = MethodType(self._forward_for_layer(index), attention)

    def _forward_for_layer(self, index):
        def forward(attention, inputs_q, *, decode_cache=None, patch_mask=None):
            if self._active is None or decode_cache is None:
                raise ValueError('Experimental native attention requires an active stepper call')
            rows, patches, _ = inputs_q.shape
            identities, starts, cuts = self._active
            if len(identities) != rows:
                raise ValueError('Native sequence metadata does not match transformer rows')
            if patch_mask is None:
                patch_mask = torch.zeros(rows, patches, dtype=torch.bool, device=inputs_q.device)
            query, key, value = [tensor.view(rows, patches, attention.num_heads, attention.head_dim)
                                 for tensor in torch.chunk(attention.qkv_proj(inputs_q), 3, dim=-1)]
            # Preserve upstream order exactly: RoPE, Q/K RMS, learned Q scaling.
            masked = patch_mask.to(torch.int32).sum(-1) + decode_cache.num_masked
            positions = (torch.arange(patches, device=inputs_q.device)[None, :]
                         + decode_cache.next_index[:, None] - masked[:, None])
            query = attention.rotary_position_embedding(query, positions)
            key = attention.rotary_position_embedding(key, positions)
            query, key = attention.query_ln(query), attention.key_ln(key)
            query = attention.per_dim_scale(query)
            sizes = [patches - cut for cut in cuts]
            dense_rows = not any(cuts)
            packed = ([tensor.reshape(rows * patches, attention.num_heads, attention.head_dim)
                       for tensor in (query, key, value)] if dense_rows else [
                           torch.cat([tensor[row, cut:] for row, cut in enumerate(cuts)]).contiguous()
                           for tensor in (query, key, value)])
            output = self.arena.attend(index, 0, *packed, identities, starts, sizes)
            if dense_rows:
                dense = output.reshape_as(query)
            else:
                dense = torch.zeros_like(query)
                cursor = 0
                for row, (cut, size) in enumerate(zip(cuts, sizes)):
                    dense[row, cut:] = output[cursor:cursor + size]
                    cursor += size
            decode_cache.next_index += patches
            decode_cache.num_masked = masked
            return attention.out(dense.reshape(rows, patches, attention.in_features)), decode_cache

        return forward

    @torch.inference_mode()
    def initialize(self, context, horizon):
        if self._closed:
            raise ValueError('Native stepper is closed')
        if (not isinstance(context, torch.Tensor) or context.ndim != 2
                or context.device != self.arena.device
                or 2 * context.shape[0] + len(self.arena.allocated) > self.arena.max_sequences):
            raise ValueError('Native inputs must fit the arena series slots on its CUDA device')
        state = super().initialize(context, horizon)
        # Drop whole masked PATCH tokens only. Trimming raw histories before
        # original outer/running normalization would change forecast semantics.
        observed = ~state.padding
        starts = observed.shape[1] - observed.sum(-1)
        expected = torch.arange(observed.shape[1], device=context.device) < starts[:, None]
        if not torch.equal(state.padding, expected) or not observed.any(-1).all():
            super().release(state)
            raise ValueError('Native proof supports finite histories with left padding only')
        masked = [int(start) // self.model.p for start in starts.tolist()]
        self._counter += 1
        identities = tuple(tuple(f'proof-{self._counter}-sign-{stream}-row-{row}'
                                 for row in range(state.rows)) for stream in range(2))
        reserved = []
        try:
            for stream_names in identities:
                for identity, cut in zip(stream_names, masked):
                    self.arena.reserve(identity, state.cache_capacity - cut)
                    reserved.append(identity)
        except BaseException:
            for identity in reserved:
                self.arena.release(identity)
            super().release(state)
            raise
        self._routes[id(state)] = (state, identities, masked)
        return state

    def sequence_ids(self, state):
        self._owner(state)
        return tuple(name for stream in self._routes[id(state)][1] for name in stream)

    def _activate(self, states, streams):
        identities, starts, cuts = [], [], []
        for stream in streams:
            for state in states:
                _, names, masked = self._routes[id(state)]
                identities.extend(names[stream])
                starts.extend(0 if not state.blocks_done else state.cache_position - cut for cut in masked)
                cuts.extend(masked if not state.blocks_done else [0] * state.rows)
        self._active = tuple(identities), tuple(starts), tuple(cuts)

    def _prefill_passes(self, states):
        self._activate(states, (0, 1))
        try:
            super()._prefill_passes(states)
        finally:
            self._active = None

    def _block_decode_forward(self, states, inputs, masks, cache_groups, positions):
        from .block_decode import block_decode_forward

        # The parent's unchanged _advance_passes fuses flip statistics, neural
        # calls and output storage. Only cache preparation/attention differ.
        self._activate(states, (0, 1))
        try:
            return block_decode_forward(self.model, inputs, masks, cache_groups, positions,
                                        attention_backend=self)
        finally:
            self._active = None

    def prepare(self, model, inputs, masks, cache_groups, positions):
        """Validate counter-only groups and prepare shared native RoPE metadata."""
        from .block_decode import _decode_positions

        if (model is not self.model or self._active is None or inputs.ndim != 3
                or inputs.shape[1:] != (model.m, model.p) or masks.shape != inputs.shape
                or not cache_groups or len(cache_groups) != len(positions)):
            raise ValueError('Native block attention requires matching active decode groups')
        groups, cursor = [], 0
        for caches, position in zip(cache_groups, positions, strict=True):
            if type(position) is not int or position < 0 or len(caches) != model.x:
                raise ValueError('Native decode groups require a physical position and every layer')
            rows = caches[0].next_index.numel()
            if rows < 1 or any(cache.next_index.shape != (rows,) or cache.num_masked.shape != (rows,)
                               or cache.key.shape != (rows, 0, model.h, model.hd)
                               or cache.value.shape != cache.key.shape for cache in caches):
                raise ValueError('Native decode requires counters with empty dense K/V')
            groups.append((slice(cursor, cursor + rows), caches, position))
            cursor += rows
        identities, starts, cuts = self._active
        if cursor != inputs.shape[0] or cursor != len(identities) or any(cuts):
            raise ValueError('Native decode routes must partition the unmasked neural batch')
        # Counter arithmetic is the exact same helper used by the dense control.
        position, num_masked, _ = _decode_positions(inputs, masks, groups)
        return groups, position, num_masked, (identities, starts, (model.m,) * cursor)

    def attend(self, layer_index, query, key, value, metadata):
        """One packed attention call per layer, including both independent signs."""
        identities, starts, sizes = metadata
        shape = query.shape
        packed = [tensor.reshape(-1, self.model.h, self.model.hd) for tensor in (query, key, value)]
        return self.arena.attend(layer_index, 0, *packed, identities, starts, sizes).reshape(shape)

    def _new_caches(self, rows, capacity, device):
        # Native KV lives in arena layer views. These are only upstream-compatible
        # counters, not hidden dense-cache allocations.
        return [DecodeCache(
            next_index=torch.zeros(rows, dtype=torch.int32, device=device),
            num_masked=torch.zeros(rows, dtype=torch.int32, device=device),
            key=torch.empty(rows, 0, self.model.h, self.model.hd, device=device),
            value=torch.empty(rows, 0, self.model.h, self.model.hd, device=device),
        ) for _ in range(self.model.x)]

    def _cohort(self, states, phase):
        states = tuple(states)
        if phase == 'prefill':
            return super()._cohort(states, phase)
        if not states or len({id(state) for state in states}) != len(states):
            raise ValueError('Native decode requires nonempty unique states')
        # The native backend uses per-row positions. Keep each state's phase,
        # ownership and completion checks, without dense-cache index equality.
        for state in states:
            super()._cohort([state], phase)
        return states

    def _free(self, state):
        route = self._routes.pop(id(state), None)
        if route is not None:
            for stream in route[1]:
                for identity in stream:
                    self.arena.release(identity)

    def _finish_block(self, states):
        super()._finish_block(states)
        for state in states:
            if self.complete(state):
                self._free(state)

    def release(self, state):
        self._owner(state)
        self._free(state)
        super().release(state)

    def close(self):
        if self._closed:
            return
        for state, _, _ in tuple(self._routes.values()):
            self.release(state)
        for attention, original in self._original:
            attention.forward = original
        self._original.clear()
        self.arena.close()
        self._active = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
