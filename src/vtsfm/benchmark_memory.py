"""Benchmark-only CUDA allocator counters for a vLLM worker extension.

Use --worker-extension-cls vtsfm.benchmark_memory.MemoryMetrics and invoke methods through
the development /collective_rpc endpoint only at benchmark point boundaries.
"""

import torch


class MemoryMetrics:
    def bench_memory_config(self):
        properties = torch.cuda.get_device_properties(self.device)
        return {
            'torch_version': str(torch.__version__),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'num_threads': torch.get_num_threads(),
            'num_interop_threads': torch.get_num_interop_threads(),
            'gpu': {
                'device': str(self.device),
                'name': properties.name,
                'uuid': str(getattr(properties, 'uuid', '')),
            },
        }

    def bench_memory_reset(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        return self.bench_memory_stats()

    def bench_memory_stats(self):
        from torch._dynamo import config, utils

        torch.cuda.synchronize(self.device)
        device = self.device
        return {
            "device": str(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "dynamo": {"unique_graphs": utils.counters.get("stats", {}).get("unique_graphs", 0),
                       "calls_captured": utils.counters.get("stats", {}).get("calls_captured", 0),
                       "frames_total": utils.counters.get("frames", {}).get("total", 0),
                       "recompile_limit": config.recompile_limit},
        }

    def bench_memory_empty_cache(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return self.bench_memory_stats()


    def bench_request_state(self):
        """Live numerical-worker request/slot counts; no allocator reset or tracing."""
        runner = self.model_runner
        state = {'worker_requests': len(runner.requests),
                 'worker_slots': runner.input_batch.num_reqs}
        executor = getattr(runner.get_model(), 'block_executor', None)
        if executor is not None:
            state.update(executor.snapshot())
        return state

    def bench_timesfm_forecast(self):
        """Experimental fixed-owner counters from the model that serves /pooling."""
        return self.model_runner.get_model().fixed_forecast_snapshot()
