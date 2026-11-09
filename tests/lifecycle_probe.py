"""Explicitly imported instrumentation for manual TimesFM lifecycle tests only.

The checkpoint, ForecastModel, pooler and IO remain production implementations.
These hooks inject deterministic faults and expose state without modifying vLLM
or adding diagnostic routes to normal vTSFM serving.
"""

import gc
import json
import os
from pathlib import Path
import time
import weakref

from fastapi import Request
from vllm.entrypoints.serve.dev.rpc.api_router import router

from vtsfm.adapters.pooling import ForecastIOProcessor
from vtsfm.adapters.transport import decode_metadata


_INSTANCES = []
_CONTEXTS = []
_METADATA = []
_ORIGINAL_INIT = ForecastIOProcessor.__init__
_ORIGINAL_PRE = ForecastIOProcessor.pre_process
_ORIGINAL_POST = ForecastIOProcessor.post_process


def _init(self, *args, **kwargs):
    _ORIGINAL_INIT(self, *args, **kwargs)
    _INSTANCES.append(weakref.ref(self))


def _pre(self, prompt, *args, **kwargs):
    prepared = _ORIGINAL_PRE(self, prompt, *args, **kwargs)
    _CONTEXTS[:] = [ref for ref in _CONTEXTS if ref() is not None]
    _CONTEXTS.append(weakref.ref(prepared['multi_modal_data']['timeseries']['context'][0]))
    _METADATA[:] = [ref for ref in _METADATA if ref() is not None]
    _METADATA.append(weakref.ref(prepared['multi_modal_data']['timeseries']['request_metadata'][0]))
    identities = [item.get('item_id', '') or '' for item in prompt[0]]
    if any(identity.startswith('__fail_pre__') for identity in identities):
        raise ValueError('injected failure after numerical preprocessing')
    if any(identity.startswith('__fail_submit__') for identity in identities):
        # Fail input validation after IO preprocessing, before EngineCore admission.
        prepared['multi_modal_data']['timeseries'].pop('request_metadata')
    return prepared


def _post(self, *args, **kwargs):
    result = _ORIGINAL_POST(self, *args, **kwargs)
    if any((row.get('item_id', '') or '').startswith('__fail_post__')
           for row in result['predictions']):
        raise ValueError('injected failure after forecast postprocessing')
    return result


def io_snapshot():
    gc.collect()
    _INSTANCES[:] = [ref for ref in _INSTANCES if ref() is not None]
    _CONTEXTS[:] = [ref for ref in _CONTEXTS if ref() is not None]
    _METADATA[:] = [ref for ref in _METADATA if ref() is not None]
    instances = [ref() for ref in _INSTANCES]
    return {'io_instances': len(instances),
            'io_pending': sum(len(getattr(instance, 'pending', {})) for instance in instances),
            'live_preprocessed_contexts': len(_CONTEXTS),
            'live_preprocessed_metadata': len(_METADATA)}


ForecastIOProcessor.__init__ = _init
ForecastIOProcessor.pre_process = _pre
ForecastIOProcessor.post_process = _post


@router.get('/lifecycle-test/state')
async def lifecycle_state(request: Request):
    state = io_snapshot()
    engine = request.app.state.engine_client
    state['api_unfinished'] = engine.output_processor.get_num_unfinished_requests()
    return state


class LifecycleProbeExtension:
    """Load-time-imported worker RPCs; no checkpoint or forecast replacement."""

    def lifecycle_install(self, directory):
        self.lifecycle_directory = str(directory)
        model = self.model_runner.get_model()
        original = model.forward
        path = Path(directory)
        path.mkdir(exist_ok=True)

        def traced_forward(*args, **kwargs):
            encoded = kwargs.get('request_metadata')
            if encoded is not None:
                metadata = [decode_metadata(value) for value in encoded]
                identities = [item or '' for request in metadata for item in request['item_ids']]
                if any(identity.startswith('__gate__') for identity in identities):
                    (path / 'entered').write_text(json.dumps({
                        'request_ids': [request['request_id'] for request in metadata],
                        'item_ids': identities, 'pid': os.getpid(),
                    }))
                    deadline = time.monotonic() + 20
                    while not (path / 'release').exists():
                        if time.monotonic() >= deadline:
                            raise RuntimeError('test gate timed out before release')
                        time.sleep(0.01)
                if any(identity.startswith('__fail_model__') for identity in identities):
                    (path / 'model-failure').write_text('injected worker model exception')
                    raise RuntimeError('injected TimesFM model-execution failure')
            return original(*args, **kwargs)

        model.forward = traced_forward
        return self.lifecycle_snapshot()

    def lifecycle_snapshot(self):
        import torch

        runner = self.model_runner
        return {'worker_cached_requests': len(runner.requests),
                'worker_slots': runner.input_batch.num_reqs,
                'worker_encoder_cache': len(runner.encoder_cache),
                'worker_pooling_params': len(runner.input_batch.pooling_params),
                'worker_pooling_states': len(runner.input_batch.pooling_states),
                'worker_has_execute_model_state': runner.execute_model_state is not None,
                'pooler_has_routes': runner.get_model().pooler.routes is not None,
                'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved()}
