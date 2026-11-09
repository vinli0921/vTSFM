"""Pinned Moirai checkpoint and pooling settings."""

import json
from vtsfm.adapters.base import AdapterSpec, FORECAST_VLLM_ARGS
from vtsfm.adapters.forecast import ForecastSettings
from vtsfm.execution import ExecutionCapabilities

MODEL_ID = 'Salesforce/moirai-2.0-R-small'
MODEL_REVISION = '30f43ff08c8494f4943ae1521e9d4e94a0fbb389'
FORECAST_SETTINGS = ForecastSettings(
    max_context=7168, max_horizon=1024,
    quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    default_horizon=96, default_quantiles=(0.1, 0.5, 0.9),
)
# From config.json at MODEL_REVISION; GPU tests check the loaded checkpoint.
PATCH_SIZE = 16
OUTPUT_BLOCK_SIZE = PATCH_SIZE * 4  # patch_size * num_predict_token per decoder pass
MOIRAI2 = AdapterSpec(
    name='moirai2', model_id=MODEL_ID, model_revision=MODEL_REVISION,
    architectures=('VTSFMMoiraiForForecasting',),
    model_class='vtsfm.adapters.moirai2.model:VTSFMMoiraiForForecasting',
    io_processor_class='vtsfm.adapters.moirai2.io_processor.MoiraiIOProcessor',
    upstream_io_plugin=None,
    native_server_module=None,
    vllm_args=FORECAST_VLLM_ARGS + (
        '--hf-overrides', json.dumps({'architectures': ['VTSFMMoiraiForForecasting'],
                                     'num_attention_heads': 6}),
    ),
    supports_group_attention=False,
    execution_capabilities=ExecutionCapabilities(
        max_context=FORECAST_SETTINGS.max_context, max_horizon=FORECAST_SETTINGS.max_horizon,
        quantiles=FORECAST_SETTINGS.quantiles, supports_point_forecast=False,
        input_patch_size=PATCH_SIZE, output_block_size=OUTPUT_BLOCK_SIZE,
        # Horizons beyond one block recurse on all 9 quantile trajectories (9x rows).
        # The runtime masks ragged padding, but keep this False until the GPU
        # ragged regression in tests/test_moirai_runtime.py passes.
    ),
)
