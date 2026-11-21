"""Pinned vLLM V1 worker bridge; imported only in a serving environment.

vLLM owns scheduling, slot compaction, execution and result association. Its
pooling metadata omits IDs, so a load-time callback exposes the current slot
order to our model. No vLLM source file or runner method is patched.
"""

import vllm
from vllm.v1.worker.gpu_worker import Worker

from vtsfm.adapters.multimodal import MODALITY
from vtsfm.adapters.transport import decode_metadata
from vtsfm.config import VLLM_VERSION


def slot_request_ids(runner):
    """Return original numerical IDs in current worker order; None for profiling."""
    if runner.input_batch.num_reqs == 0:
        return None
    result = []
    for internal_id in runner.input_batch.req_ids:
        state = runner.requests[internal_id]
        features = [feature for feature in state.mm_features if feature.modality == MODALITY]
        if len(features) != 1 or features[0].data is None:
            raise ValueError('Each forecast slot must retain one numerical payload')
        if state.num_prompt_tokens != 1:
            raise ValueError('Whole-forecast routing requires one placeholder token per request')
        result.append(decode_metadata(features[0].data['request_metadata'].data)['request_id'])
    return tuple(result)


class ForecastWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        if vllm.__version__ != VLLM_VERSION or self.use_v2_model_runner:
            raise ValueError(f'Forecast routing requires pinned vLLM {VLLM_VERSION} Model Runner V1')
        if (not self.model_config.enforce_eager or self.parallel_config.pipeline_parallel_size != 1
                or self.parallel_config.tensor_parallel_size != 1):
            raise ValueError('Forecast routing requires eager execution on one GPU')
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.model_runner.get_model().set_request_order(lambda: slot_request_ids(self.model_runner))

    def compile_or_warm_up_model(self):
        times = super().compile_or_warm_up_model()
        warm = getattr(self.model_runner.get_model(), 'warm_fixed_forecast', None)
        if warm is not None:
            warm()
        return times

    def shutdown(self):
        model = getattr(getattr(self, 'model_runner', None), 'model', None)
        close = getattr(model, 'close_fixed_forecast', None)
        try:
            if close is not None:
                close()
        finally:
            super().shutdown()
