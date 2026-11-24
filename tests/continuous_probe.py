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

    def continuous_watch_prefill_storage(self):
        """Keep weak storage handles so the test can observe actual release across slot reuse."""
        from torch.multiprocessing.reductions import StorageWeakRef

        executor = self.model_runner.get_model().block_executor
        stepper = executor.stepper
        prefill = stepper.prefill
        self._owner_storages = {}
        self._shared_owners = {}
        self._admissions = []
        self._disjoint = True

        def watch(states):
            names = {id(state): executor.metadata[key].wire_id.rsplit(':', 1)[0]
                     for key, state in executor.states.items()}
            self._admissions.append({'new': [names[id(s)] for s in states],
                                     'live_storages': self._live_cache_storages()})
            result = prefill(states)
            if len(states) > 1:
                seen = set()
                for state in states:
                    name = names[id(state)]
                    owner = self._owner_storages[name] = set()
                    counts, kv_counts = [], []
                    logical = backing = 0
                    for stream in state.passes:
                        storages = {}
                        kv_storages = set()
                        for layer in stream.caches:
                            for t in (layer.key, layer.value, layer.next_index, layer.num_masked):
                                storage = t.untyped_storage()
                                logical += t.numel() * t.element_size()
                                handle = StorageWeakRef(storage)
                                storages[handle] = storage.nbytes()
                                if t is layer.key or t is layer.value:
                                    kv_storages.add(handle)
                        self._disjoint &= seen.isdisjoint(storages)
                        seen.update(storages)
                        owner.update(storages)
                        counts.append(len(storages))
                        kv_counts.append(len(kv_storages))
                        backing += sum(storages.values())
                    self._shared_owners[name] = {
                        'counts_by_flip': counts, 'kv_counts_by_flip': kv_counts,
                        'backing_bytes': backing, 'logical_bytes': logical}
            return result

        stepper.prefill = watch
        return True

    def _live_cache_storages(self):
        return {name: sum(not handle.expired() for handle in handles)
                for name, handles in self._owner_storages.items()}

    def continuous_storage_report(self):
        report = {'shared_owners': self._shared_owners, 'disjoint': self._disjoint,
                  'admissions': self._admissions, 'live_storages': self._live_cache_storages()}
        self._shared_owners = {}
        self._owner_storages = {}
        self._admissions = []
        self._disjoint = True
        return report
