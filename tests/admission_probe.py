"""Explicit test-only worker diagnostics for TimesFM admission checks."""

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
