"""Explicit test-only worker diagnostics for numerical admission checks."""

import json
from pathlib import Path

from vtsfm.adapters.transport import decode_metadata
from vtsfm.benchmark_memory import MemoryMetrics


class AdmissionProbeExtension(MemoryMetrics):
    """Retain production forecasts; record the actual scheduled worker envelope."""

    def admission_install(self, path):
        trace = Path(path)
        model = self.model_runner.get_model()
        original = model.forward

        def forward(*args, **kwargs):
            encoded = kwargs.get('request_metadata')
            if encoded is not None:
                requests = [decode_metadata(value) for value in encoded]
                if requests and all(request['request_id'] != 'vtsfm-dummy' for request in requests):
                    with trace.open('a') as output:
                        output.write(json.dumps({'request_ids': [request['request_id'] for request in requests],
                            'rows': [len(request['item_ids']) for request in requests],
                            'contexts': [max(request['context_lengths']) for request in requests],
                            'horizons': [request['horizon'] for request in requests]}) + '\n')
            return original(*args, **kwargs)

        model.forward = forward
        return {'worker_trace': str(trace)}

    def admission_snapshot(self):
        """Inspect request-owned state, separately from upstream model caches."""
        runner = self.model_runner
        return {
            'worker_requests': len(runner.requests),
            'worker_slots': runner.input_batch.num_reqs,
            'worker_encoder_cache': len(runner.encoder_cache),
            'worker_pooling_params': len(runner.input_batch.pooling_params),
            'worker_pooling_states': len(runner.input_batch.pooling_states),
            'worker_execute_state': runner.execute_model_state is not None,
            'pooler_routes': runner.get_model().pooler.routes is not None,
            **self.bench_memory_stats(),
        }
