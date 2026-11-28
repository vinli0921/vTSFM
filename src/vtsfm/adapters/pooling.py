"""Whole-forecast pooling, following the Chronos plugin's execution pattern."""

import os
import uuid

import numpy as np
import torch
from torch import nn
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import attn_type
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.utils import length_from_prompt_token_ids_or_embeds

from vtsfm.config import FORECAST_BATCH_ROWS_ENV, MAX_FORECAST_BATCH_ROWS
from vtsfm.admission import forward_resident_bytes, resolve_adapter_limits

from .execution_batch import execute_scheduled
from .forecast import ForecastSettings, format_predictions, parse_request
from .multimodal import MODALITY
from .result_transport import encode_result_metadata, pack_result, unpack_result
from .transport import decode_requests, encode_metadata
from .routing import order_requests, validate_forecast


class ForecastPooler(Pooler):
    """Crop padded forecasts and attach their small, self-describing IO metadata.

    Forward and pooling run consecutively in the pinned eager V1 worker. Only
    CPU metadata is retained between them; pooling consumes it once, including
    on failure. GPU outputs live in the returned tensor, never in a pending dict.
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
        return [pack_result(hidden_states[index, :rows, :, :horizon], metadata)
                for index, (rows, horizon, metadata) in enumerate(routes)]


# Attention stays inside the upstream model; vLLM does not manage its KV cache.
@attn_type('attention_free')
class ForecastModel(nn.Module, IsAttentionFree, SupportsMultiModal):
    """Runtime owns `model` and `forecast(context, horizon)`; context is left-NaN-padded
    [series, time], and output is [series, adapter quantiles, horizon]."""
    supports_multimodal_raw_input_only = True
    is_pooling_model = True
    packed_modules_mapping = {}
    batch_spec = None
    admission_cost = None

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        from vtsfm.stepping import CONTINUOUS_ENV
        if os.environ.get(CONTINUOUS_ENV) == '1' and (self.batch_spec is None or self.batch_spec.name != 'timesfm25'):
            raise ValueError('Forecast-block execution currently supports only TimesFM')
        self.forecast_limits = resolve_adapter_limits(self.batch_spec, os.environ)
        if self.forecast_limits is not None:
            limits = self.forecast_limits
            # Check direct LLM/vllm launches too, before loading the checkpoint.
            slots = getattr(getattr(vllm_config, 'scheduler_config', None), 'max_num_seqs', 1)
            limits.validate_launch(slots, self.admission_cost(
                limits.max_request_rows, limits.max_context, limits.max_horizon))
        self.runtime = self.runtime_class(vllm_config.model_config.hf_config.vtsfm_config)
        self.model = self.runtime.model
        self.pooler = ForecastPooler()
        self.request_order = None
        self.forecast_batch_rows = int(os.environ.get(FORECAST_BATCH_ROWS_ENV, '0'))
        if self.forecast_batch_rows and (self.batch_spec is None
                or not 2 <= self.forecast_batch_rows <= MAX_FORECAST_BATCH_ROWS):
            raise ValueError('Unsupported adapter or row cap for experimental forecast batching')
        self.forecast_batch_trace = os.environ.get('VTSFM_FORECAST_BATCH_TRACE')

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
        executor = getattr(self, 'block_executor', None)
        if executor is not None:
            return executor.forward(input_ids, positions, inputs_embeds=inputs_embeds, **kwargs)
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
        limits = getattr(self, 'forecast_limits', None)
        if limits is not None:
            costs = [self.admission_cost(len(request.item_ids), request.context.shape[1], request.horizon)
                     for request in requests]
            # Frontend admission + startup slot checks guarantee these bounds.
            # These assertions protect internal callers, not normal overload handling.
            for cost in costs:
                limits.validate_request(cost)
            if len(requests) > limits.max_scheduled_requests or forward_resident_bytes(costs) > limits.max_forward_bytes:
                raise ValueError('Scheduled forecast inputs exceed the configured resident envelope')
        if self.forecast_batch_rows and len(requests) > 1:
            forecasts = execute_scheduled(self.runtime, self.batch_spec, requests,
                                          self.forecast_batch_rows, self.forecast_batch_trace,
                                          limits=limits, cost_for_shape=self.admission_cost)
        else:
            forecasts = (self.runtime.forecast(request.context, request.horizon) for request in requests)
        outputs = []
        for request, forecast in zip(requests, forecasts, strict=True):
            # Keep the shared output boundary explicit for each tensor runtime.
            output = forecast.to(request.context.device)
            validate_forecast(request, output, len(self.settings.quantiles))
            outputs.append(output)
        rows = max(len(request.item_ids) for request in requests)
        horizon = max(request.horizon for request in requests)
        packed = outputs[0].new_zeros((length, rows, len(self.settings.quantiles), horizon))
        for index, (request, output) in enumerate(zip(requests, outputs)):
            packed[index, :len(request.item_ids), :, :request.horizon] = output
        self.pooler.routes = tuple((len(request.item_ids), request.horizon,
            encode_result_metadata(request.item_ids, request.horizon, request.quantile_levels))
            for request in requests)
        return packed

    def load_weights(self, weights):
        self.model.load_state_dict(dict(weights), strict=True)
        return {name for name, _ in self.named_parameters()}


class ForecastIOProcessor(IOProcessor):
    settings: ForecastSettings
    admission_spec = None
    admission_cost = None

    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        from vtsfm.stepping import CONTINUOUS_ENV
        if os.environ.get(CONTINUOUS_ENV) == '1' and (self.admission_spec is None or self.admission_spec.name != 'timesfm25'):
            raise ValueError('Forecast-block IO currently supports only TimesFM')
        self.forecast_limits = resolve_adapter_limits(self.admission_spec, os.environ)

    def _check_limits(self, inputs, horizon):
        if self.forecast_limits is not None:
            self.forecast_limits.validate_request(self.admission_cost(
                len(inputs), max(len(item['target']) for item in inputs), horizon))

    def parse_data(self, data):
        parsed = parse_request(data, self.settings)
        self._check_limits(parsed[0], parsed[1])
        return parsed

    def pre_process(self, prompt, request_id=None, **kwargs):
        inputs, horizon, levels = prompt
        self._check_limits(inputs, horizon)
        item_ids = [item.get('item_id') for item in inputs]
        # Reject oversized output metadata here, before submitting GPU work.
        encode_result_metadata(item_ids, horizon, levels)
        # parse_request supplies FP32 arrays, so asarray is a no-op there. Row
        # copies stay in NumPy; one from_numpy call avoids torch.tensor(list).
        targets = [np.asarray(item['target'], dtype=np.float32) for item in inputs]
        context = np.full((len(targets), max(map(len, targets))), np.nan, dtype=np.float32)
        for row, target in zip(context, targets):
            row[len(row) - len(target):] = target
        context = torch.from_numpy(context)
        # HTTP headers can repeat and offline calls omit IDs. Each numerical
        # payload therefore needs an invocation ID independent of either case.
        identity = f'{request_id or "vtsfm"}:{uuid.uuid4().hex}'
        metadata = encode_metadata(identity, item_ids,
            [len(item['target']) for item in inputs], horizon, levels)
        from vtsfm.stepping import CONTINUOUS_ENV
        blocks = (horizon + 127) // 128 if os.environ.get(CONTINUOUS_ENV) == '1' else 1
        return {'prompt_token_ids': [1] * blocks, 'multi_modal_data': {
            MODALITY: {'context': [context],
                       'request_metadata': [torch.tensor(list(metadata), dtype=torch.uint8)]}}}

    def post_process(self, model_output, request_id=None, **kwargs):
        if len(model_output) != 1:
            raise ValueError('Expected one whole-forecast output')
        output, item_ids, horizon, levels = unpack_result(model_output[0].outputs.data)
        if horizon > self.settings.max_horizon or any(q not in self.settings.quantiles for q in levels):
            raise ValueError('Forecast result metadata does not match adapter settings')
        # vLLM wraps plugin data with the HTTP request ID and creation time.
        return format_predictions(output, item_ids, levels, horizon, self.settings)
