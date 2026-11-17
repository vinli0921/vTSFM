# Copyright 2026 vTSFM contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-shape complete forecasts, eager or one CUDA graph.

Finite equal-length contiguous CUDA FP32 inputs only. The optimized stepper owns
all numerical operations; these private subclasses freeze cache geometry and
remove prefill scalar readbacks. No serving flag selects this implementation.
"""

import time

import torch

from .block_decode import _decode_positions, block_decode_forward
from .native_stepping import PagedTimesFMStepper
from .paged_cache import PagedCacheArena
from .stepping import TimesFMStepper


class _FrozenArena(PagedCacheArena):
    """Hold every full-forecast metadata tensor at a fixed address."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plans = {}

    def add_plan(self, identities, start, queries):
        starts, sizes = (start,) * len(identities), (queries,) * len(identities)
        cumulative = tuple(i * queries for i in range(len(identities) + 1))
        self.plans[identities, starts, sizes] = super()._metadata(
            identities, starts, sizes, cumulative, (start + queries,) * len(identities))

    def _metadata(self, identities, starts, sizes, cumulative, lengths):
        try:
            return self.plans[identities, starts, sizes]
        except KeyError as error:
            raise ValueError('Full-forecast metadata must be frozen before capture') from error

    def close(self):
        self.plans.clear()
        super().close()


class _DenseStepper(TimesFMStepper):
    def _prefill_forward(self, inputs, masks, caches):
        output, groups = block_decode_forward(self.model, inputs, masks, [caches], [0], prefill=True)
        return output, groups[0]


class _PagedStepper(PagedTimesFMStepper):
    _arena_type = _FrozenArena

    def __init__(self, runtime, rows, patches, horizon):
        capacity = patches + 4 * ((horizon - 1) // 128)
        super().__init__(runtime, max_sequences=rows, max_tokens=capacity)
        try:
            # All prefills now use the shared explicit helper. Restore the
            # native model immediately; it is never monkeypatched for replay.
            for attention, original in self._original:
                attention.forward = original
            self._original.clear()
            self.identities = tuple(tuple(f'graph-sign-{sign}-row-{row}' for row in range(rows))
                                    for sign in range(2))
            for names in self.identities:
                for identity in names:
                    self.arena.reserve(identity, capacity)
                self.arena.add_plan(names, 0, patches)
            all_names = self.identities[0] + self.identities[1]
            for block in range((horizon - 1) // 128):
                self.arena.add_plan(all_names, patches + block * 4, 4)
        except BaseException:
            self.close()
            raise

    def initialize(self, context, horizon):
        # CPU-only append bookkeeping is used while tracing/eager execution.
        # Replay instead reruns captured GPU counter initialization and overwrites
        # every valid KV prefix before it is read. Leases remain pinned.
        for lease in self.arena._leases.values():
            lease.positions.clear()
        state = TimesFMStepper.initialize(self, context, horizon)
        self._routes[id(state)] = (state, self.identities, [0] * state.rows)
        return state

    def _prefill_forward(self, inputs, masks, caches):
        output, groups = block_decode_forward(self.model, inputs, masks, [caches], [0],
                                              prefill=True, attention_backend=self)
        return output, groups[0]

    def prepare(self, model, inputs, masks, cache_groups, positions):
        if (model is not self.model or self._active is None or inputs.ndim != 3
                or inputs.shape[-1] != model.p or inputs.shape[1] < 1 or masks.shape != inputs.shape
                or not cache_groups or len(cache_groups) != len(positions)):
            raise ValueError('Invalid frozen full-forecast attention geometry')
        groups, cursor = [], 0
        for caches, position in zip(cache_groups, positions, strict=True):
            if len(caches) != model.x or type(position) is not int or position < 0:
                raise ValueError('Invalid frozen cache group')
            rows = caches[0].next_index.numel()
            if rows < 1 or any(cache.next_index.shape != (rows,) or cache.num_masked.shape != (rows,)
                               or cache.key.shape != (rows, 0, model.h, model.hd)
                               or cache.value.shape != cache.key.shape for cache in caches):
                raise ValueError('Frozen full-forecast attention requires counters with empty dense K/V')
            groups.append((slice(cursor, cursor + rows), caches, position))
            cursor += rows
        identities, starts, cuts = self._active
        if cursor != inputs.shape[0] or cursor != len(identities) or any(cuts):
            raise ValueError('Full-forecast graphs require fixed finite history layouts')
        position, num_masked, _ = _decode_positions(inputs, masks, groups)
        return groups, position, num_masked, (identities, starts, (inputs.shape[1],) * cursor)

    def _free(self, state):
        # Completion retires only the Python trace state. Graph leases cannot be
        # freed/reused until FixedForecast destroys the graph and closes arena.
        self._routes.pop(id(state), None)


class FixedForecast:
    """One fixed B/C/H numerical forecast, with independent owned outputs.

    ``run`` includes finite-input validation, device-to-device input staging and
    output cloning in both modes. Shape, horizon, cache plans and leases are
    frozen at construction. The graph contains outer normalization, both sign
    prefills, all continuation blocks, both heads and quantile postprocessing.
    Capture/setup and persistent slot ownership are outside steady-state timing.
    """

    @torch.inference_mode()
    def __init__(self, runtime, example, horizon, *, backend='dense', capture=True, warmups=3):
        if backend not in ('dense', 'paged') or type(capture) is not bool:
            raise ValueError('Require dense/paged backend and boolean capture')
        if (not isinstance(example, torch.Tensor) or example.ndim != 2 or example.dtype != torch.float32
                or example.device.type != 'cuda' or not 1 <= example.shape[0] <= 16
                or not 1 <= example.shape[1] <= 8192 or not example.is_contiguous()
                or not torch.isfinite(example).all()):
            raise ValueError('Require finite nonempty contiguous CUDA FP32 example [rows,context]')
        if type(horizon) is not int or not 1 <= horizon <= 1024 or type(warmups) is not int or warmups < 1:
            raise ValueError('Require horizon 1..1024 and at least one warmup')
        if next(runtime.model.parameters()).device != example.device:
            raise ValueError('Example and model must share a CUDA device')
        self.runtime, self.backend, self.horizon = runtime, backend, horizon
        self.shape, self.device = tuple(example.shape), example.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.graph = self.stepper = self.static_output = self.input = None
        self.replay_count = 0
        self._closed = False
        self.setup = {'backend': backend, 'capture': capture, 'shape': self.shape, 'horizon': horizon,
                      'warmups': warmups, 'scope': 'Fixed finite equal-length contiguous CUDA FP32 forecasts; no HTTP/runner integration. Both modes include input validation/copy and owned output clone. Cache leases/plans fixed for owner lifetime.'}
        started = time.perf_counter()
        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        needs_sync = True
        try:
            with torch.cuda.stream(self.stream):
                self.input = example.clone()
                patches = (example.shape[1] + 31) // 32
                self.stepper = (_DenseStepper(runtime) if backend == 'dense' else
                                _PagedStepper(runtime, example.shape[0], patches, horizon))
                warm_start = time.perf_counter()
                for _ in range(warmups):
                    self.static_output = self._body()
                self.stream.synchronize()
                # Warmups were the last executed work. Capture only records it.
                needs_sync = False
                self.setup['warmup_seconds'] = time.perf_counter() - warm_start
                self.setup['fixed_pool_bytes'] = self.stepper.arena.cache_bytes if backend == 'paged' else 0
                capture_start = time.perf_counter()
                if capture:
                    self._capture()
                    self.stream.synchronize()
                self.setup['capture_seconds'] = time.perf_counter() - capture_start if capture else 0.
            self.setup['construction_seconds'] = time.perf_counter() - started
            self.setup['allocated_bytes_after_setup'] = torch.cuda.memory_allocated(self.device)
            self.setup['reserved_bytes_after_setup'] = torch.cuda.memory_reserved(self.device)
        except BaseException as error:
            try:
                self._close(synchronize=needs_sync)
            except BaseException as cleanup_error:
                error.add_note(f'Forecast setup cleanup also failed: {cleanup_error!r}')
            raise

    def _capture(self):
        graph = torch.cuda.CUDAGraph()
        body_error = None
        try:
            with torch.cuda.graph(graph, stream=self.stream):
                try:
                    self.static_output = self._body()
                except BaseException as error:
                    # CUDA still has to end capture, which can raise on its own.
                    body_error = error
        except BaseException as error:
            graph = None
            if body_error is None:
                raise
            body_error.add_note(f'Ending CUDA graph capture also failed: {error!r}')
        else:
            self.graph = graph
        if body_error is not None:
            raise body_error

    def _body(self):
        state = self.stepper.initialize(self.input, self.horizon)
        try:
            self.stepper.prefill([state])
            while not self.stepper.complete(state):
                self.stepper.advance([state])
            return self.stepper.finalize(state)
        finally:
            self.stepper.release(state)

    @torch.inference_mode()
    def run(self, context):
        if self._closed:
            raise ValueError('Full-forecast owner is closed')
        if (not isinstance(context, torch.Tensor) or tuple(context.shape) != self.shape
                or context.dtype != torch.float32 or context.device != self.device):
            raise ValueError('Full-forecast input shape/device/dtype requires a new owner')
        if not context.is_contiguous():
            raise ValueError('Full-forecast inputs must be contiguous')
        if not torch.isfinite(context).all():
            raise ValueError('Full-forecast inputs must be finite; mask changes need a new owner')
        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            context.record_stream(self.stream)
            self.input.copy_(context)
            if self.graph is None:
                self.static_output = self._body()
            else:
                self.graph.replay()
                self.replay_count += 1
            output = self.static_output.clone()
        caller.wait_stream(self.stream)
        output.record_stream(caller)
        return output

    def close(self):
        self._close(synchronize=True)

    def _close(self, *, synchronize):
        if self._closed:
            return
        if synchronize:
            self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            if self.graph is not None:
                self.graph.reset()
                self.graph = None
            self.static_output = self.input = None
            if self.stepper is not None and self.backend == 'paged':
                self.stepper.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
