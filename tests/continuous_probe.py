"""Explicit test-only block gates and state snapshots for the real worker."""

from pathlib import Path
import time

from vtsfm.benchmark_memory import MemoryMetrics


class ContinuousProbeExtension(MemoryMetrics):
    def continuous_install(self, directory):
        path = Path(directory)
        path.mkdir(exist_ok=True)
        model = self.model_runner.get_model()

        def gate(kind):
            if not (path / f'{kind}-armed').exists() or (path / f'{kind}-entered').exists():
                return
            (path / f'{kind}-entered').write_text('worker reached block boundary')
            deadline = time.monotonic() + 20
            while not (path / f'{kind}-release').exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{kind} test gate was not released within 20 seconds')
                time.sleep(.01)

        for name, kind in (('prefill', 'prefill'), ('advance', 'decode')):
            original = getattr(model.stepper, name)

            def wrapped(*args, _original=original, _kind=kind, **kwargs):
                result = _original(*args, **kwargs)
                gate(_kind)
                return result

            setattr(model.stepper, name, wrapped)
        return self.continuous_snapshot()

    def continuous_snapshot(self):
        import torch

        runner = self.model_runner
        state = runner.get_model().block_executor.snapshot()
        return {**state, 'worker_requests': len(runner.requests),
                'worker_slots': runner.input_batch.num_reqs,
                'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved()}
