"""Whole-forecast pooling, following the Chronos plugin's execution pattern."""

import time

import torch
from torch import nn
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import attn_type
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds

from .forecast import ForecastSettings, format_predictions, parse_request
from .multimodal import MODALITY


# Attention stays inside the upstream model; vLLM does not manage its KV cache.
@attn_type('attention_free')
class ForecastModel(nn.Module, IsAttentionFree, SupportsMultiModal):
    """Runtime owns `model` and `forecast(context, horizon)`; context is left-NaN-padded
    [series, time], and output is [series, adapter quantiles, horizon]."""
    supports_multimodal_raw_input_only = True
    is_pooling_model = True
    packed_modules_mapping = {}

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        self.runtime = self.runtime_class(vllm_config.model_config.hf_config.vtsfm_config)
        self.model = self.runtime.model
        self.pooler = IdentityPooler()

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality != MODALITY:
            raise ValueError(f'Unsupported modality: {modality}')
        return None

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, **kwargs):
        return torch.empty((input_ids.shape[0], 0), device=input_ids.device)

    @torch.inference_mode()
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        length = length_from_prompt_token_ids_or_embeds(input_ids, inputs_embeds)
        if kwargs.get('context') is None:
            return torch.zeros(length, 0, device=positions.device)
        output = self.runtime.forecast(kwargs['context'], int(kwargs['horizon']))
        return output.to(positions.device).unsqueeze(0).expand(length, *(-1 for _ in output.shape))

    def load_weights(self, weights):
        self.model.load_state_dict(dict(weights), strict=True)
        return {name for name, _ in self.named_parameters()}


class ForecastIOProcessor(IOProcessor):
    settings: ForecastSettings

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.pending = {}

    def parse_request(self, request):
        return parse_request(request.data if hasattr(request, 'data') else request, self.settings)

    def pre_process(self, prompt, request_id=None, **kwargs):
        inputs, horizon, levels = prompt
        context = torch.full((len(inputs), max(len(x['target']) for x in inputs)), float('nan'))
        for row, item in zip(context, inputs):
            row[-len(item['target']):] = torch.tensor(item['target'], dtype=torch.float32)
        self.pending[request_id] = ([item.get('item_id') for item in inputs], horizon, levels)
        return {'prompt_token_ids': [1], 'multi_modal_data': {
            MODALITY: {'context': context, 'horizon': horizon}}}

    def post_process(self, model_output, request_id=None, **kwargs):
        item_ids, horizon, levels = self.pending.pop(request_id)
        if len(model_output) != 1:
            raise ValueError('Expected one whole-forecast output')
        output = model_output[0].outputs.data
        if output.ndim == 4 and output.shape[0] == 1:
            output = output.squeeze(0)
        result = format_predictions(output, item_ids, levels, horizon, self.settings)
        result['request_id'] = request_id
        return result

    def validate_or_generate_params(self, params=None):
        return PoolingParams(task='plugin')

    def output_to_response(self, plugin_output):
        return IOProcessorResponse(request_id=plugin_output['request_id'], created_at=int(time.time()),
                                   data={'predictions': plugin_output['predictions']})
