"""Pinned TimesFM checkpoint and pooling settings."""

import json
from vtsfm.adapters.base import AdapterSpec, FORECAST_VLLM_ARGS
from vtsfm.adapters.forecast import ForecastSettings

MODEL_ID = 'google/timesfm-2.5-200m-pytorch'
MODEL_REVISION = '1d952420fba87f3c6dee4f240de0f1a0fbc790e3'
FORECAST_SETTINGS = ForecastSettings(
    max_context=8192, max_horizon=1024,
    quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    default_horizon=96, default_quantiles=(0.1, 0.5, 0.9),
)
TIMESFM25 = AdapterSpec(
    name='timesfm25', model_id=MODEL_ID, model_revision=MODEL_REVISION,
    architectures=('VTSFMTimesFMForForecasting',),
    model_class='vtsfm.adapters.timesfm25.model:VTSFMTimesFMForForecasting',
    io_processor_class='vtsfm.adapters.timesfm25.io_processor.TimesFMIOProcessor',
    upstream_io_plugin=None,
    native_server_module=None,
    vllm_args=FORECAST_VLLM_ARGS + (
        '--hf-overrides', json.dumps({'architectures': ['VTSFMTimesFMForForecasting']}),
    ),
    supports_group_attention=False,
)
