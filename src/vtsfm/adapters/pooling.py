"""Whole-forecast pooling, following the Chronos plugin's execution pattern."""

import uuid

import torch
from torch import nn
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import attn_type
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.utils import length_from_prompt_token_ids_or_embeds

from .forecast import ForecastSettings, format_predictions, parse_request
from .multimodal import MODALITY
from .transport import decode_requests, encode_metadata
from .routing import order_requests, validate_forecast


class ForecastPooler(Pooler):
    """Crop padded worker rows to variable-size per-request [series, Q, H].

    Forward and pooling run consecutively in the pinned eager V1 worker. Only
    shapes are retained between them; pooling consumes these once, including on
    failure. GPU outputs live in the returned tensor, never in a pending dict.
    """

    def __init__(self):
        super().__init__()
        self.routes = None
        self.profiling = True

    def get_supported_tasks(self):
        return {'plugin'}

    def forward(self, hidden_states, pooling_metadata):
        routes, self.routes = self.routes, None
        if routes is None:
            if not self.profiling:
                raise ValueError('No unconsumed forecast outputs are available for pooling')
            # vLLM's synthetic startup/profile calls have no live request slots.
            return [hidden_states.new_zeros((1, 9, 1)) for _ in pooling_metadata.pooling_params]
        if len(routes) != len(pooling_metadata.pooling_params) or hidden_states.ndim != 4:
            raise ValueError('Forecast outputs do not match worker pooling slots')
        cursor = pooling_metadata.get_pooling_cursor()
        if (cursor.num_scheduled_tokens_cpu.tolist() != [1] * len(routes)
                or pooling_metadata.prompt_lens.tolist() != [1] * len(routes)):
            raise ValueError('Whole-forecast pooling requires complete one-token requests')
        if hidden_states.shape[0] != len(routes):
            raise ValueError('Forecast output row count does not match pooling requests')
        return [hidden_states[index, :rows, :, :horizon]
                for index, (rows, horizon) in enumerate(routes)]


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
        self.pooler = ForecastPooler()
        self.request_order = None

    def set_request_order(self, callback):
        self.request_order = callback

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
        self.pooler.routes = None
        self.pooler.profiling = False
        callback = getattr(self, 'request_order', None)
        slot_ids = callback() if callback is not None else None
        if kwargs.get('context') is None:
            if slot_ids:
                raise ValueError('Scheduled forecast slots are missing numerical payloads')
            self.pooler.profiling = True
            return torch.zeros(length, 0, device=positions.device)
        if callback is not None and slot_ids is None:
            self.pooler.profiling = True
            return torch.zeros(length, 0, device=positions.device)
        requests = decode_requests(kwargs['context'], kwargs['request_metadata'])
        if callback is None and len(requests) != 1:
            raise ValueError('Multi-request forecasts require the vTSFM routing worker')
        if slot_ids is not None:
            requests = order_requests(requests, slot_ids)
        if length != len(requests):
            if all(request.request_id == 'vtsfm-dummy' for request in requests):
                self.pooler.profiling = True
                return torch.zeros(length, 1, len(self.settings.quantiles), 1, device=positions.device)
            raise ValueError('Forecast placeholders do not match worker request count')
        outputs = []
        for request in requests:
            # Moirai's current wrapper returns CPU tensors via its NumPy API.
            # Tensor runtimes already return on the execution device.
            output = self.runtime.forecast(request.context, request.horizon).to(request.context.device)
            validate_forecast(request, output, len(self.settings.quantiles))
            outputs.append(output)
        rows = max(len(request.item_ids) for request in requests)
        horizon = max(request.horizon for request in requests)
        packed = outputs[0].new_zeros((length, rows, len(self.settings.quantiles), horizon))
        for index, (request, output) in enumerate(zip(requests, outputs)):
            packed[index, :len(request.item_ids), :, :request.horizon] = output
        self.pooler.routes = tuple((len(request.item_ids), request.horizon) for request in requests)
        return packed

    def load_weights(self, weights):
        self.model.load_state_dict(dict(weights), strict=True)
        return {name for name, _ in self.named_parameters()}


class ForecastIOProcessor(IOProcessor):
    settings: ForecastSettings

    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        self.pending = {}

    def parse_data(self, data):
        return parse_request(data, self.settings)

    def pre_process(self, prompt, request_id=None, **kwargs):
        inputs, horizon, levels = prompt
        context = torch.full((len(inputs), max(len(x['target']) for x in inputs)), float('nan'))
        for row, item in zip(context, inputs):
            row[-len(item['target']):] = torch.tensor(item['target'], dtype=torch.float32)
        self.pending[request_id] = ([item.get('item_id') for item in inputs], horizon, levels)
        metadata = encode_metadata(request_id or f'vtsfm-{uuid.uuid4().hex}',
            [item.get('item_id') for item in inputs], [len(item['target']) for item in inputs], horizon, levels)
        return {'prompt_token_ids': [1], 'multi_modal_data': {
            MODALITY: {'context': [context],
                       'request_metadata': [torch.tensor(list(metadata), dtype=torch.uint8)]}}}

    def post_process(self, model_output, request_id=None, **kwargs):
        item_ids, horizon, levels = self.pending.pop(request_id)
        if len(model_output) != 1:
            raise ValueError('Expected one whole-forecast output')
        output = model_output[0].outputs.data
        if output.ndim == 4 and output.shape[0] == 1:
            output = output.squeeze(0)
        # vLLM wraps plugin data with the HTTP request ID and creation time.
        return format_predictions(output, item_ids, levels, horizon, self.settings)
