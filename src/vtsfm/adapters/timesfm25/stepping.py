# Copyright 2026 vTSFM contributors
#
# Decoder progression is derived from Google LLC's TimesFM 2.5 Torch decoder
# (Copyright 2025 Google LLC), distributed under the Apache License, Version 2.0.
# See https://github.com/google-research/timesfm and the repository LICENSE.
"""Resumable TimesFM forecasts with adapter-owned, request-local decode caches.

Shared decode calls batch neural layers while preserving each request's owned
caches. Temporary attention gathers never replace those caches, and retired
requests retain no peer storage.
"""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from timesfm.torch.util import DecodeCache, revin, update_running_stats


# W4: 20 layers × K/V × 2 rows × 44 positions × 16 heads × 80 FP32 values, per flip.
# Larger slabs increased CUDA allocator reservation at the four-request admission limit.
_MAX_PACKED_PREFILL_KV_BYTES = 18_022_400


@dataclass
class _PassState:
    caches: list = field(default_factory=list)
    last_n: object = None
    last_mu: object = None
    last_sigma: object = None
    feedback: object = None
    spread: object = None

    def clear(self):
        self.caches.clear()
        self.last_n = self.last_mu = self.last_sigma = None
        self.feedback = self.spread = None


@dataclass
class TimesFMForecastState:
    """CPU geometry plus owned CUDA tensors; never shared between requests."""

    rows: int
    context_length: int
    horizon: int
    padded_context_length: int
    cache_capacity: int
    owner: object = field(repr=False)
    blocks_done: int = 0
    released: bool = False
    normalized: object = field(default=None, repr=False)
    padding: object = field(default=None, repr=False)
    mean: object = field(default=None, repr=False)
    std: object = field(default=None, repr=False)
    positive: object = field(default=None, repr=False)
    output: object = field(default=None, repr=False)
    passes: list = field(default_factory=lambda: [_PassState(), _PassState()], repr=False)

    @property
    def cache_position(self):
        return self.padded_context_length // 32 + 4 * (self.blocks_done - 1) if self.blocks_done else 0

    @property
    def bucket(self):
        if self.released:
            raise ValueError('Released forecast state cannot be scheduled')
        return ('prefill', self.padded_context_length) if self.blocks_done == 0 else ('decode',)

    @property
    def cache_bytes(self):
        return sum(tensor.numel() * tensor.element_size()
                   for stream in self.passes for cache in stream.caches
                   for tensor in (cache.key, cache.value))

    @property
    def tensor_bytes(self):
        buffers = [self.normalized, self.padding, self.mean, self.std, self.positive, self.output]
        for stream in self.passes:
            buffers.extend((stream.last_n, stream.last_mu, stream.last_sigma,
                            stream.feedback, stream.spread))
            for cache in stream.caches:
                buffers.extend((cache.next_index, cache.num_masked, cache.key, cache.value))
        return sum(tensor.numel() * tensor.element_size() for tensor in buffers if tensor is not None)


def _split_prefill_caches(caches, rows, capacity):
    """Copy one owner's prefill rows into storage private to this flip."""
    key = caches[0].key
    _, width, heads, head_dim = key.shape
    owner_width = min(capacity, width)
    elements = (rows.stop - rows.start) * owner_width * heads * head_dim
    if 2 * len(caches) * elements * key.element_size() > _MAX_PACKED_PREFILL_KV_BYTES:
        return [DecodeCache(next_index=cache.next_index[rows].clone(),
                            num_masked=cache.num_masked[rows].clone(),
                            key=cache.key[rows, :owner_width].clone(),
                            value=cache.value[rows, :owner_width].clone()) for cache in caches]
    kv = [tensor[rows] for cache in caches for tensor in (cache.key, cache.value)]
    if owner_width < width:
        # Mixed horizons share the widest prefill cache. Retain only this owner's capacity.
        kv = [tensor[:, :owner_width] for tensor in kv]
    # Each layer gets disjoint contiguous views. Native prefill can produce
    # int64 mask counts while its write indices remain int32.
    kv = torch.stack(kv).unbind()
    next_indices = torch.stack([cache.next_index[rows] for cache in caches]).unbind()
    mask_counts = torch.stack([cache.num_masked[rows] for cache in caches]).unbind()
    return [DecodeCache(key=kv[2 * i], value=kv[2 * i + 1],
                        next_index=next_indices[i], num_masked=mask_counts[i])
            for i in range(len(caches))]


class TimesFMStepper:
    """Advance original TimesFM algorithms one 128-point forecast block at a time.

    Inputs are FP32 tensors ``[series, raw padded history]`` with left NaNs for
    missing history. States and forecasts stay on the model device. Normal and
    sign-flipped decoders retain independent running stats, feedback and caches.
    Prefill and AR calls batch both passes, so the internal neural batch has twice the
    independent series counted by admission; the work bound includes both flips.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.model = runtime.model
        self.owner = object()
        if (self.model.p, self.model.o, self.model.os, self.model.q, self.model.aridx) != (32, 128, 1024, 10, 5):
            raise ValueError('Block execution requires the pinned TimesFM 2.5 architecture')
        from .block_decode import validate_block_decode_source

        validate_block_decode_source(self.model)

    @torch.inference_mode()
    def initialize(self, context, horizon, *, context_lengths=None):
        if (not isinstance(context, torch.Tensor) or context.ndim != 2
                or not context.shape[0] or not 1 <= context.shape[1] <= 8192):
            raise ValueError('TimesFM block context must be nonempty [series, history<=8192]')
        if type(horizon) is not int or not 1 <= horizon <= self.model.os:
            raise ValueError('TimesFM block horizon must be an integer in [1,1024]')
        device = next(self.model.parameters()).device
        values = context.to(device=device, dtype=torch.float32)
        observed = torch.isfinite(values)
        left = -values.shape[1] % self.model.p
        values = F.pad(values.masked_fill(~observed, 0), (left, 0))
        padding = F.pad(~observed, (left, 0), value=True)
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True)
        rows, width = values.shape
        return TimesFMForecastState(
            rows, context.shape[1], horizon, width,
            width // self.model.p + self.model.m * ((horizon - 1) // self.model.o), self.owner,
            normalized=revin(values, mean, std, reverse=False), padding=padding,
            mean=mean, std=std, positive=(values >= 0).all(dim=-1, keepdim=True).unsqueeze(-1),
            output=torch.empty((rows, 9, horizon), dtype=torch.float32, device=device),
        )

    def complete(self, state):
        self._owner(state)
        return state.blocks_done >= (state.horizon + self.model.o - 1) // self.model.o

    def model_call_work(self, states):
        """Conservative shared-call work bound, including both flips."""
        states = tuple(states)
        return 2 * sum(state.rows for state in states) * max(state.cache_capacity for state in states) ** 2 if states else 0

    def _owner(self, state):
        if not isinstance(state, TimesFMForecastState) or state.owner is not self.owner:
            raise ValueError('Forecast state belongs to a different TimesFM stepper')

    def _cohort(self, states, phase):
        states = tuple(states)
        if not states or len({id(state) for state in states}) != len(states):
            raise ValueError('Block execution needs nonempty unique forecast states')
        for state in states:
            self._owner(state)
            if state.released or self.complete(state):
                raise ValueError('Completed or released forecast state cannot advance')
        if any(state.bucket[0] != phase or state.bucket != states[0].bucket for state in states):
            raise ValueError('Shared TimesFM block states require matching execution buckets')
        if phase == 'decode' and any(state.cache_position + self.model.m > state.cache_capacity
                                     for state in states):
            raise ValueError('TimesFM block write exceeds the request cache capacity')
        return states

    def _new_caches(self, rows, capacity, device):
        return [DecodeCache(
            next_index=torch.zeros(rows, dtype=torch.int32, device=device),
            num_masked=torch.zeros(rows, dtype=torch.int32, device=device),
            key=torch.zeros(rows, capacity, self.model.h, self.model.hd, dtype=torch.float32, device=device),
            value=torch.zeros(rows, capacity, self.model.h, self.model.hd, dtype=torch.float32, device=device),
        ) for _ in range(self.model.x)]

    def _store(self, states, stream, caches, stats, block, spread=None):
        cursor = 0
        for state in states:
            section = slice(cursor, cursor + state.rows)
            target = state.passes[stream]
            if caches is not None and len(states) == 1:
                # New prefill allocations and singleton updates already belong
                # exclusively to this request. Keep their storage in place.
                target.caches = caches
            elif caches is not None:
                target.caches = _split_prefill_caches(caches, section, state.cache_capacity)
            target.last_n, target.last_mu, target.last_sigma = (value[section].clone() for value in stats)
            target.feedback = block[section, :, self.model.aridx].clone()
            if spread is not None:
                target.spread = spread[section, :state.horizon, 1:].clone()
            cursor += state.rows

    def _prefill_passes(self, states):
        # One stream-major [x; -x] call, as in _advance_passes. Rows (and both
        # flip streams) stay independent; each stream then stores its own half.
        normalized = torch.cat([state.normalized for state in states])
        masks = torch.cat([state.padding for state in states])
        rows = normalized.shape[0]
        normalized = torch.cat((normalized, -normalized))
        masks = masks.repeat(2, 1)
        inputs = normalized.reshape(2 * rows, -1, self.model.p)
        masks = masks.reshape(2 * rows, -1, self.model.p)
        n = torch.zeros(2 * rows, device=normalized.device)
        mu = torch.zeros_like(n)
        sigma = torch.zeros_like(n)
        means, sigmas = [], []
        for patch in range(inputs.shape[1]):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, inputs[:, patch], masks[:, patch])
            means.append(mu)
            sigmas.append(sigma)
        context_mu, context_sigma = torch.stack(means, 1), torch.stack(sigmas, 1)
        caches = self._new_caches(2 * rows, max(state.cache_capacity for state in states), normalized.device)
        inputs = revin(inputs, context_mu, context_sigma, reverse=False)
        inputs = torch.where(masks, 0.0, inputs)
        (_, _, point, spread), caches = self._prefill_forward(inputs, masks, caches)
        point = revin(point, context_mu, context_sigma, reverse=True).reshape(2 * rows, -1, self.model.o, self.model.q)[:, -1]
        spread = revin(spread, context_mu, context_sigma, reverse=True).reshape(2 * rows, -1, self.model.os, self.model.q)[:, -1]
        for stream in range(2):
            half = slice(stream * rows, (stream + 1) * rows)
            # Row slices are views. A singleton request keeps both halves of its
            # own allocation; _store splits shared cohorts into per-request storage.
            stream_caches = [DecodeCache(
                next_index=cache.next_index[half], num_masked=cache.num_masked[half],
                key=cache.key[half], value=cache.value[half],
            ) for cache in caches]
            self._store(states, stream, stream_caches, (n[half], mu[half], sigma[half]), point[half], spread[half])

    def _prefill_forward(self, inputs, masks, caches):
        """Preserve native prefill; fixed-shape graph experiments override it."""
        return self.model(inputs, masks, caches)

    def _advance_passes(self, states):
        # Stream-major order keeps normal and flipped statistics independent.
        streams = [state.passes[stream] for stream in range(2) for state in states]
        rows = sum(state.rows for state in states)
        inputs = torch.cat([stream.feedback for stream in streams]).reshape(-1, self.model.m, self.model.p)
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        n, mu, sigma = (torch.cat([getattr(stream, name) for stream in streams])
                        for name in ('last_n', 'last_mu', 'last_sigma'))
        means, sigmas = [], []
        for patch in range(self.model.m):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, inputs[:, patch], masks[:, patch])
            means.append(mu)
            sigmas.append(sigma)
        context_mu, context_sigma = torch.stack(means, 1), torch.stack(sigmas, 1)
        inputs = revin(inputs, context_mu, context_sigma, reverse=False)
        (_, _, point, _), _ = self._block_decode_forward(
            states, inputs, masks, [stream.caches for stream in streams],
            [state.cache_position for _ in range(2) for state in states],
        )
        point = revin(point, context_mu, context_sigma, reverse=True).reshape(-1, self.model.m, self.model.o, self.model.q)[:, -1]
        for stream in range(2):
            section = slice(stream * rows, (stream + 1) * rows)
            self._store(states, stream, None, tuple(value[section] for value in (n, mu, sigma)), point[section])

    def _block_decode_forward(self, states, inputs, masks, cache_groups, positions):
        """Keep the optimized executor shared with experimental attention paths."""
        from .block_decode import block_decode_forward

        return block_decode_forward(self.model, inputs, masks, cache_groups, positions)

    def _finish_block(self, states):
        for state in states:
            first, flipped = state.passes
            start = state.blocks_done * self.model.o
            stop = min(start + self.model.o, state.horizon)
            length = stop - start
            # Each flip already owns its raw point-head median for the next AR
            # step. Stage their difference in the unpublished output's median row.
            torch.sub(first.feedback[:, :length], flipped.feedback[:, :length], out=state.output[:, 4, start:stop])
            state.blocks_done += 1
            state.normalized = state.padding = None
            if not self.complete(state):
                continue

            median = state.output[:, 4, :].unsqueeze(-1) / 2
            spread = (first.spread - flipped.spread.flip(-1)) / 2
            for stream in state.passes:
                stream.clear()
            continuous = spread - spread[..., 4:5] + median
            lower = torch.cat((median, continuous[..., :4].flip(-1)), dim=-1).cummin(-1).values[..., 1:].flip(-1)
            upper = torch.cat((median, continuous[..., 5:]), dim=-1).cummax(-1).values[..., 1:]
            quantiles = revin(torch.cat((lower, median, upper), -1), state.mean, state.std, reverse=True)
            quantiles = torch.where(state.positive, quantiles.clamp_min(0), quantiles)
            state.output.copy_(quantiles.transpose(1, 2))

    @torch.inference_mode()
    def prefill(self, states):
        states = self._cohort(states, 'prefill')
        self._prefill_passes(states)
        self._finish_block(states)

    @torch.inference_mode()
    def advance(self, states):
        states = self._cohort(states, 'decode')
        self._advance_passes(states)
        self._finish_block(states)

    def finalize(self, state):
        self._owner(state)
        if state.released or not self.complete(state):
            raise ValueError('Only a completed, unreleased TimesFM forecast can finalize')
        return state.output

    def release(self, state):
        self._owner(state)
        for stream in state.passes:
            stream.clear()
        state.passes.clear()
        state.normalized = state.padding = state.mean = state.std = None
        state.positive = state.output = None
        state.released = True
