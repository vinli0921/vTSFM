"""A single FP32 CUDA graph for Toto's whole-forecast vLLM worker."""

from threading import Lock
from time import perf_counter

import torch

from .spec import DECODE_BLOCK_SIZE


class TotoForecastGraph:
    def __init__(self, shape):
        self.shape = shape
        self.lock = Lock()
        self.graph = None
        self.setup = None
        self.decode_cache = None
        self.graph_replays = self.fallbacks = self.capture_attempts = 0

    @torch.inference_mode()
    def warm(self, runtime):
        with self.lock:
            if self.graph is not None:
                return
            parameter = next(runtime.model.parameters())
            device = parameter.device
            if device.type != 'cuda' or parameter.dtype != torch.float32:
                raise ValueError('Toto CUDA graph replay requires CUDA FP32 weights')

            if self.shape[-1] <= DECODE_BLOCK_SIZE:
                self._capture(runtime, device)
                return
            # Multi-block forecasts reuse model-owned KV buffers. The graph
            # holds captured pointers even when an eager fallback needs a new
            # cache, so retain the graph cache separately until graph teardown.
            model = runtime.model
            eager_cache, eager_key = model._kv_cache, model._kv_cache_key
            model._kv_cache = model._kv_cache_key = None
            try:
                self._capture(runtime, device)
                self.decode_cache = model._kv_cache
            finally:
                model._kv_cache, model._kv_cache_key = eager_cache, eager_key

    def _capture(self, runtime, device):
        self.capture_attempts += 1
        started = perf_counter()
        rows, width, horizon = self.shape
        with torch.cuda.device(device):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            reserved_before = torch.cuda.memory_reserved(device)
            allocated_before = torch.cuda.memory_allocated(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                context = torch.zeros((rows, width), dtype=torch.float32, device=device)
                observed = torch.ones((rows, width), dtype=torch.bool, device=device)
                for _ in range(3):
                    runtime._forecast(context, observed, horizon)
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = runtime._forecast(context, observed, horizon)
            stream.synchronize()
            setup = {
                'seconds': perf_counter() - started,
                'free_bytes_before': free_bytes,
                'allocated_delta_bytes': torch.cuda.memory_allocated(device) - allocated_before,
                'reserved_delta_bytes': torch.cuda.memory_reserved(device) - reserved_before,
            }
        # Publish only a complete capture. A failed startup retains no owner buffers.
        self.device, self.stream = device, stream
        self.context, self.observed, self.output = context, observed, output
        self.graph, self.setup = graph, setup

    def try_forecast(self, context, observed, horizon):
        with self.lock:
            if (*context.shape, horizon) != self.shape:
                self.fallbacks += 1
                return None
            if self.graph is None:
                raise RuntimeError('Toto CUDA graph was not initialized by the forecast worker')
            if (context.device != self.device or observed.device != self.device
                    or context.dtype != torch.float32 or observed.dtype != torch.bool
                    or observed.shape != context.shape):
                raise ValueError('Toto CUDA graph requires matching FP32 values and bool mask on its device')

            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                context.record_stream(self.stream)
                observed.record_stream(self.stream)
                self.context.copy_(context)
                self.observed.copy_(observed)
                self.graph.replay()
                # Pooling can outlive this call. The next replay will overwrite the static output.
                output = self.output.clone()
            caller.wait_stream(self.stream)
            output.record_stream(caller)
            self.graph_replays += 1
            return output

    def snapshot(self):
        with self.lock:
            return {
                'mode': 'cuda-graph', 'shape': self.shape, 'setup': self.setup,
                'capture_attempts': self.capture_attempts, 'graph_replays': self.graph_replays,
                'fallbacks': self.fallbacks,
            }

    def close(self):
        with self.lock:
            if self.graph is not None:
                self.stream.synchronize()
                self.graph.reset()
                self.graph = self.context = self.observed = self.output = self.decode_cache = None
