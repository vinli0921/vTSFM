"""Test-only, checkpoint-free model for the real vLLM transport probe.

Importing this module registers the concrete model and its production multimodal
processor. The worker-extension hook imports it in vLLM's spawned GPU worker as
well as the driver. It is never registered by vTSFM's normal plugin.
"""

import hashlib
import json

import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.multimodal import MULTIMODAL_REGISTRY

from vtsfm.adapters.multimodal import (
    ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo,
)
from vtsfm.adapters.pooling import ForecastModel
from vtsfm.adapters.transport import decode_requests


ARCHITECTURE = 'VTSFMTransportProbe'
DIGEST_WIDTH = 32
OUTPUT_WIDTH = 2 + DIGEST_WIDTH


def fingerprint(request):
    """Hash all numerical values and semantic metadata, including ragged padding.

    Copying the context to the CPU is intentional probe instrumentation, not a
    proposed forecast execution path. Digest bytes are exactly representable in
    FP32, so assertions do not need numerical tolerances.
    """
    metadata = json.dumps({
        'request_id': request.request_id,
        'item_ids': request.item_ids,
        'context_lengths': request.context_lengths,
        'horizon': request.horizon,
        'quantile_levels': request.quantile_levels,
        'shape': list(request.context.shape),
    }, sort_keys=True, separators=(',', ':')).encode()
    values = request.context.detach().cpu().contiguous().numpy().tobytes()
    return list(hashlib.sha256(metadata + values).digest())


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo,
    dummy_inputs=ForecastInputBuilder,
)
class ProbeModel(ForecastModel):
    """Return transport fingerprints rather than model forecasts.

    One token represents one request. Equal invocation counters and a batch size
    greater than one in the returned rows prove a common worker forward call.
    This test-specific fixed-width output is not the forecast routing in #7.
    """

    def __init__(self, *, vllm_config, prefix=''):
        nn.Module.__init__(self)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.pooler = IdentityPooler()
        self.invocations = 0

    @torch.inference_mode()
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        length = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if kwargs.get('context') is None:
            return torch.zeros(length, OUTPUT_WIDTH, device=positions.device)
        metadata = kwargs['request_metadata']
        if all(json.loads(bytes(row.tolist()))['request_id'] == 'vtsfm-dummy'
               for row in metadata):
            return torch.zeros(length, OUTPUT_WIDTH, device=positions.device)
        requests = decode_requests(kwargs['context'], kwargs['request_metadata'])
        assert all(request.context.device == positions.device for request in requests), (
            'Numerical contexts did not reach the worker execution device')
        # Profiling warmup pads dummy prompts to the maximum sequence length.
        # Only actual one-token requests constitute a transport observation.
        if length != len(requests):
            return torch.zeros(length, OUTPUT_WIDTH, device=positions.device)
        self.invocations += 1
        rows = [[self.invocations, len(requests), *fingerprint(request)]
                for request in requests]
        return torch.tensor(rows, dtype=torch.float32, device=positions.device)

    def load_weights(self, weights):
        raise AssertionError('The transport probe must use load_format="dummy"')


class ProbeWorkerExtension:
    """Import-only worker extension; it adds no worker behavior or patches."""


ModelRegistry.register_model(ARCHITECTURE, 'transport_probe:ProbeModel')
