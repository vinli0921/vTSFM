"""Single fixed-shape owner for the experimental TimesFM pooling path."""

import os
import threading

import torch

from .graph_config import FORECAST_GRAPH_SHAPE


class ServingFixedForecast:
    def __init__(self, mode):
        self.mode = mode
        self.backend, execution = mode.split('-')
        self.capture = execution == 'graph'
        self.owner = None
        self.setup = None
        self.calls = self.fallbacks = self.capture_attempts = self.graph_replays = 0
        self.lock = threading.Lock()

    def try_forecast(self, context, horizon):
        with self.lock:
            # Ragged histories are represented by left NaNs. They need the
            # ordinary decoder even if the padded width matches the owner.
            if ((*context.shape, horizon) != FORECAST_GRAPH_SHAPE
                    or not torch.isfinite(context).all()):
                self.fallbacks += 1
                return None
            if self.owner is None:
                raise RuntimeError('TimesFM fixed forecast owner was not initialized by the routing worker')
            before = self.owner.replay_count
            output = self.owner.run(context.contiguous(), prevalidated=True)
            self.graph_replays += self.owner.replay_count - before
            self.calls += 1
            return output

    @torch.inference_mode()
    def warm(self, runtime):
        with self.lock:
            if self.owner is None:
                self._create(runtime)

    def _create(self, runtime):
        from .full_graph import FixedForecast

        parameter = next(runtime.model.parameters())
        if (parameter.device.type != 'cuda' or parameter.dtype != torch.float32
                or os.environ.get('NVIDIA_TF32_OVERRIDE') != '0'):
            raise ValueError('TimesFM forecast mode requires CUDA FP32 with TF32 disabled')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Keep both backends within the same one-owner memory policy. Check
        # after vLLM startup, when the worker's actual residency is known.
        free_bytes, _ = torch.cuda.mem_get_info(parameter.device)
        if free_bytes < 512 * 1024**2:
            raise RuntimeError('TimesFM fixed forecast setup requires at least 512 MiB free GPU memory')
        self.capture_attempts += int(self.capture)
        rows, context_length, horizon = FORECAST_GRAPH_SHAPE
        context = torch.zeros((rows, context_length), dtype=parameter.dtype, device=parameter.device)
        self.owner = FixedForecast(runtime, context, horizon, backend=self.backend, capture=self.capture)
        self.setup = dict(self.owner.setup, free_bytes_before=free_bytes)

    def snapshot(self):
        with self.lock:
            arena = self.owner.stepper.arena if self.owner is not None and self.backend == 'paged' else None
            return {'mode': self.mode, 'calls': self.calls, 'fallbacks': self.fallbacks,
                    'graph_replays': self.graph_replays, 'capture_attempts': self.capture_attempts,
                    'setup': self.setup, 'fixed_pool_bytes': arena.cache_bytes if arena else 0,
                    'leased_cache_bytes': arena.leased_cache_bytes if arena else 0}

    def close(self):
        with self.lock:
            if self.owner is not None:
                self.owner.close()
                self.owner = None
