# Copyright 2026 vTSFM contributors
#
# Decoder progression is derived from Google LLC's TimesFM 2.5 Torch decoder
# (Copyright 2025 Google LLC), distributed under the Apache License, Version 2.0.
# See https://github.com/google-research/timesfm and the repository LICENSE.
"""Resumable TimesFM forecasts with adapter-owned, request-local decode caches.

Mixed-position decode calls use row-aware writes with native positions and
masks. Request caches persist between blocks; shared calls write back only new
entries, so retired requests retain no peer storage.
"""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from timesfm.torch.util import DecodeCache, revin, update_running_stats


@dataclass
class _PassState:
    caches: list = field(default_factory=list)
    last_n: object = None
    last_mu: object = None
    last_sigma: object = None
    feedback: object = None
    spread: object = None
    block: object = None

    def clear(self):
        self.caches.clear()
        self.last_n = self.last_mu = self.last_sigma = None
        self.feedback = self.spread = self.block = None


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
                            stream.feedback, stream.spread, stream.block))
            for cache in stream.caches:
                buffers.extend((cache.next_index, cache.num_masked, cache.key, cache.value))
        return sum(tensor.numel() * tensor.element_size() for tensor in buffers if tensor is not None)


class TimesFMStepper:
    """Advance original TimesFM algorithms one 128-point forecast block at a time.

    Inputs are FP32 tensors ``[series, raw padded history]`` with left NaNs for
    missing history. States and forecasts stay on the model device. Normal and
    sign-flipped decoders retain independent running stats, feedback and caches.
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
    def initialize(self, context, horizon):
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
        """Conservative work for padded shared caches, including both flips."""
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

    @staticmethod
    def _pad_cache(values, capacity):
        extra = capacity - values.shape[1]
        return F.pad(values, (0, 0, 0, 0, 0, extra)) if extra else values

    def _merge_caches(self, states, stream, capacity):
        if len(states) == 1:
            return states[0].passes[stream].caches
        result = []
        for layer in range(self.model.x):
            parts = [state.passes[stream].caches[layer] for state in states]
            result.append(DecodeCache(
                next_index=torch.cat([part.next_index for part in parts]),
                num_masked=torch.cat([part.num_masked for part in parts]),
                key=torch.cat([self._pad_cache(part.key, capacity) for part in parts]),
                value=torch.cat([self._pad_cache(part.value, capacity) for part in parts]),
            ))
        return result

    def _store(self, states, stream, caches, stats, block, spread=None, *, positions=None):
        cursor = 0
        for number, state in enumerate(states):
            section = slice(cursor, cursor + state.rows)
            target = state.passes[stream]
            if len(states) == 1:
                # New prefill allocations and singleton updates already belong
                # exclusively to this request. Keep their storage in place.
                target.caches = caches
            elif positions is None:
                # Split prefill once; no request may retain peer allocations.
                target.caches = [DecodeCache(
                    next_index=cache.next_index[section].clone(),
                    num_masked=cache.num_masked[section].clone(),
                    key=cache.key[section, :state.cache_capacity].clone(),
                    value=cache.value[section, :state.cache_capacity].clone(),
                ) for cache in caches]
            else:
                start = positions[number]
                stop = start + self.model.m
                for owned, packed in zip(target.caches, caches, strict=True):
                    owned.next_index.copy_(packed.next_index[section])
                    owned.num_masked.copy_(packed.num_masked[section])
                    owned.key[:, start:stop].copy_(packed.key[section, start:stop])
                    owned.value[:, start:stop].copy_(packed.value[section, start:stop])
            target.last_n, target.last_mu, target.last_sigma = (value[section].clone() for value in stats)
            target.feedback = block[section, :, self.model.aridx].clone()
            target.block = block[section, :, 1:].clone()
            if spread is not None:
                target.spread = spread[section, :, 1:].clone()
            cursor += state.rows

    def _prefill_pass(self, states, stream):
        normalized = torch.cat([state.normalized for state in states])
        if stream:
            normalized = -normalized
        masks = torch.cat([state.padding for state in states])
        rows = normalized.shape[0]
        inputs = normalized.reshape(rows, -1, self.model.p)
        masks = masks.reshape(rows, -1, self.model.p)
        n = torch.zeros(rows, device=normalized.device)
        mu = torch.zeros_like(n)
        sigma = torch.zeros_like(n)
        means, sigmas = [], []
        for patch in range(inputs.shape[1]):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, inputs[:, patch], masks[:, patch])
            means.append(mu)
            sigmas.append(sigma)
        context_mu, context_sigma = torch.stack(means, 1), torch.stack(sigmas, 1)
        caches = self._new_caches(rows, max(state.cache_capacity for state in states), normalized.device)
        inputs = revin(inputs, context_mu, context_sigma, reverse=False)
        inputs = torch.where(masks, 0.0, inputs)
        (_, _, point, spread), caches = self.model(inputs, masks, caches)
        point = revin(point, context_mu, context_sigma, reverse=True).reshape(rows, -1, self.model.o, self.model.q)[:, -1]
        spread = revin(spread, context_mu, context_sigma, reverse=True).reshape(rows, -1, self.model.os, self.model.q)[:, -1]
        self._store(states, stream, caches, (n, mu, sigma), point, spread)

    def _advance_pass(self, states, stream):
        inputs = torch.cat([state.passes[stream].feedback for state in states]).reshape(-1, self.model.m, self.model.p)
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        n, mu, sigma = (torch.cat([getattr(state.passes[stream], name) for state in states])
                        for name in ('last_n', 'last_mu', 'last_sigma'))
        means, sigmas = [], []
        for patch in range(self.model.m):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, inputs[:, patch], masks[:, patch])
            means.append(mu)
            sigmas.append(sigma)
        context_mu, context_sigma = torch.stack(means, 1), torch.stack(sigmas, 1)
        caches = self._merge_caches(states, stream, max(state.cache_capacity for state in states))
        inputs = revin(inputs, context_mu, context_sigma, reverse=False)
        positions = [state.cache_position for state in states]
        if len(set(positions)) == 1:
            (_, _, point, _), caches = self.model(inputs, masks, caches)
        else:
            from .block_decode import block_decode_forward

            (_, _, point, _), caches = block_decode_forward(self.model, inputs, masks, caches)
        point = revin(point, context_mu, context_sigma, reverse=True).reshape(-1, self.model.m, self.model.o, self.model.q)[:, -1]
        self._store(states, stream, caches, (n, mu, sigma), point, positions=positions)

    def _finish_block(self, states):
        for state in states:
            first, flipped = state.passes
            start = state.blocks_done * self.model.o
            stop = min(start + self.model.o, state.horizon)
            length = stop - start
            forecast = (first.block[:, :length] - flipped.block[:, :length].flip(-1)) / 2
            spread = (first.spread[:, start:stop] - flipped.spread[:, start:stop].flip(-1)) / 2
            median = forecast[..., 4:5]
            continuous = spread - spread[..., 4:5] + median
            lower = torch.cat((median, continuous[..., :4].flip(-1)), dim=-1).cummin(-1).values[..., 1:].flip(-1)
            upper = torch.cat((median, continuous[..., 5:]), dim=-1).cummax(-1).values[..., 1:]
            quantiles = revin(torch.cat((lower, median, upper), -1), state.mean, state.std, reverse=True)
            quantiles = torch.where(state.positive, quantiles.clamp_min(0), quantiles)
            state.output[:, :, start:stop] = quantiles.transpose(1, 2)
            state.blocks_done += 1
            state.normalized = state.padding = None
            first.block = flipped.block = None
            if self.complete(state):
                for stream in state.passes:
                    stream.clear()

    @torch.inference_mode()
    def prefill(self, states):
        states = self._cohort(states, 'prefill')
        self._prefill_pass(states, 0)
        self._prefill_pass(states, 1)
        self._finish_block(states)

    @torch.inference_mode()
    def advance(self, states):
        states = self._cohort(states, 'decode')
        self._advance_pass(states, 0)
        self._advance_pass(states, 1)
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
