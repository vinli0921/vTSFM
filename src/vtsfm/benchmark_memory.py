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
        torch.cuda.synchronize(self.device)
        device = self.device
        return {
            "device": str(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    def bench_memory_empty_cache(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return self.bench_memory_stats()
