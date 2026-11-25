"""Untimed probe for the TimesFM compilation path used by the serving worker."""

import inspect

from vtsfm.benchmark_memory import MemoryMetrics


class CompileMetrics(MemoryMetrics):
    def bench_timesfm_compile_scope(self):
        runtime = self.model_runner.get_model().runtime
        targets = []
        if hasattr(runtime.model.forward, '_torchdynamo_orig_callable'):
            targets.append(getattr(runtime, 'compile_scope', 'core'))
        if hasattr(runtime._forecast, '_torchdynamo_orig_callable'):
            targets.append('whole')
        staged = getattr(runtime, 'positioned_forecast', None)
        if staged is not None and hasattr(staged._prefill_forward, '_torchdynamo_orig_callable'):
            targets.append('positioned')
        return {'compile_targets': targets, 'scope': runtime.compile_scope, 'runtime_file': inspect.getfile(type(runtime))}
