"""Pinned vLLM 0.33 V1 lifecycle/slot bridge for forecast-block execution.

GPUModelRunner updates cached request progress before forwarding and removes
finished IDs even on zero-token housekeeping steps. This worker exposes that
CPU metadata and releases the adapter's state before upstream removes it.
"""

from vtsfm.adapters.multimodal import MODALITY
from vtsfm.adapters.transport import decode_metadata
from vtsfm.block_pooling import BlockSlot
from vtsfm.worker import ForecastWorker
from vtsfm.stepping import forecast_blocks


def block_slots(runner):
    """Read scheduled slots in actual worker order; ``None`` means profiling.

    Numerical payloads and metadata live in cached mm_features. Contexts are
    transported only on the first step; ongoing steps need IDs/progress only.
    ``num_computed_tokens`` counts completed virtual forecast-block markers.
    """
    if runner.input_batch.num_reqs == 0:
        return None
    slots = []
    for request_id in runner.input_batch.req_ids:
        state = runner.requests[request_id]
        features = [feature for feature in state.mm_features if feature.modality == MODALITY]
        if len(features) != 1 or features[0].data is None:
            raise ValueError('Each forecast block slot must retain one numerical payload')
        metadata = decode_metadata(features[0].data['request_metadata'].data)
        blocks = forecast_blocks(metadata['horizon'])
        if (state.num_prompt_tokens != blocks
                or not 0 <= state.num_computed_tokens < blocks):
            raise ValueError('Forecast block markers/progress do not match the numerical horizon')
        slots.append(BlockSlot(request_id=request_id, wire_id=metadata['request_id'],
                               progress=state.num_computed_tokens, blocks=blocks))
    return tuple(slots)


class ForecastBlockWorker(ForecastWorker):
    """Keep vLLM's normal execution path; add bounded adapter-state lifecycle."""

    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.get_model()
        if getattr(model, 'block_executor', None) is None:
            raise ValueError('ForecastBlockWorker requires an adapter block executor')
        model.block_executor.set_slots(lambda: block_slots(self.model_runner))

    def execute_model(self, scheduler_output):
        # EngineCore calls the worker for finished-only scheduler output too.
        # Consume these IDs before GPUModelRunner pops its cached requests.
        self.model_runner.get_model().block_executor.forget(scheduler_output.finished_req_ids)
        return super().execute_model(scheduler_output)
