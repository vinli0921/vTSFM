"""Pinned Toto checkpoint and pooling settings."""

import json
from vtsfm.adapters.base import AdapterSpec, FORECAST_VLLM_ARGS
from vtsfm.adapters.forecast import ForecastSettings
from vtsfm.execution import ExecutionCapabilities
from .admission import DEFAULT_LIMITS

MODEL_ID = 'Datadog/Toto-2.0-313m'
MODEL_REVISION = 'a7bab288f5e95f8606f8306f86659357e1c001ef'
FORECAST_SETTINGS = ForecastSettings(
    max_context=8192, max_horizon=1024,
    quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    default_horizon=96, default_quantiles=(0.1, 0.5, 0.9),
)
DECODE_BLOCK_SIZE = 768
PATCH_SIZE = 32  # From config.json at MODEL_REVISION; GPU tests check the loaded model.
TOTO2 = AdapterSpec(
    name='toto2', model_id=MODEL_ID, model_revision=MODEL_REVISION,
    architectures=('VTSFMTotoForForecasting',),
    model_class='vtsfm.adapters.toto2.model:VTSFMTotoForForecasting',
    io_processor_class='vtsfm.adapters.toto2.io_processor.TotoIOProcessor',
    upstream_io_plugin=None,
    native_server_module=None,
    vllm_args=FORECAST_VLLM_ARGS + (
        '--hf-overrides', json.dumps({'architectures': ['VTSFMTotoForForecasting']}),
    ),
    supports_group_attention=True,
    admission_limits=DEFAULT_LIMITS,
    admission_cost='vtsfm.adapters.toto2.admission:forecast_cost',
    execution_capabilities=ExecutionCapabilities(
        max_context=FORECAST_SETTINGS.max_context, max_horizon=FORECAST_SETTINGS.max_horizon,
        quantiles=FORECAST_SETTINGS.quantiles, input_patch_size=PATCH_SIZE,
        output_block_size=DECODE_BLOCK_SIZE, supports_point_forecast=False,
        execution_options=(('decode_block_size', str(DECODE_BLOCK_SIZE)),),
    ),
)
