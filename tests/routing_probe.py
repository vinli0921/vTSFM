"""Test-only deterministic runtime using production forecast routing and pooling."""

import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from vllm import ModelRegistry
from vllm.multimodal import MULTIMODAL_REGISTRY

from vtsfm.adapters.multimodal import (
    ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo,
)
from vtsfm.adapters.pooling import ForecastModel
from vtsfm.adapters.timesfm25.spec import FORECAST_SETTINGS


ARCHITECTURE = 'VTSFMRoutingProbe'


class DeterministicRuntime:
    def __init__(self, config):
        self.model = nn.Linear(1, 1)

    def forecast(self, context, horizon):
        # Deliberately vary request duration. This is correctness instrumentation,
        # not a serving optimization or a performance workload.
        if float(context[0, -1].item()) == 31:
            time.sleep(0.03)
        quantiles = torch.arange(9, dtype=context.dtype, device=context.device)[None, :, None] * 100
        steps = torch.arange(horizon, dtype=context.dtype, device=context.device)[None, None, :]
        output = context[:, -1, None, None] + quantiles + steps
        # Exercise the CPU-output compatibility path used by the current
        # TimesFM/Moirai bring-up wrappers, alongside device-resident outputs.
        return output.cpu() if horizon == 5 else output


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo,
    dummy_inputs=ForecastInputBuilder,
)
class RoutingProbeModel(ForecastModel):
    runtime_class = DeterministicRuntime
    settings = FORECAST_SETTINGS

    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        metadata = kwargs.get('request_metadata')
        if metadata is not None and len(metadata) > 1 and not any(
                json.loads(bytes(row.tolist()))['request_id'] == 'vtsfm-dummy' for row in metadata):
            # Test-only perturbation: wire order must differ from actual worker
            # slots. Production code must recover identity, not zip by arrival.
            kwargs = dict(kwargs)
            for name in ('context', 'request_metadata'):
                value = kwargs[name]
                kwargs[name] = value.flip(0) if isinstance(value, torch.Tensor) else list(reversed(value))
        # All numerical execution and routing stay in the production superclass.
        output = super().forward(*args, **kwargs)
        path = os.environ.get('VTSFM_ROUTING_TRACE')
        metadata = kwargs.get('request_metadata')
        if path and metadata is not None:
            requests = [json.loads(bytes(row.tolist())) for row in metadata]
            if not any(request['request_id'] == 'vtsfm-dummy' for request in requests):
                event = {'pid': os.getpid(), 'request_ids': [x['request_id'] for x in requests],
                         'slot_ids': list(self.request_order()),
                         'rows': [len(x['item_ids']) for x in requests],
                         'horizons': [x['horizon'] for x in requests]}
                with Path(path).open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
        return output

    def load_weights(self, weights):
        raise AssertionError('The routing probe must use load_format="dummy"')


class RoutingProbeExtension:
    """Import-only hook that registers the test architecture in spawned workers."""


ModelRegistry.register_model(ARCHITECTURE, 'routing_probe:RoutingProbeModel')
