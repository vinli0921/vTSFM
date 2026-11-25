"""Pinned TimesFM checkpoint and pooling settings."""

import json
from vtsfm.adapters.base import AdapterSpec, FORECAST_VLLM_ARGS
from vtsfm.adapters.forecast import ForecastSettings
from vtsfm.execution import ExecutionCapabilities
from .admission import DEFAULT_LIMITS

MODEL_ID = 'google/timesfm-2.5-200m-pytorch'
MODEL_REVISION = '1d952420fba87f3c6dee4f240de0f1a0fbc790e3'
FORECAST_SETTINGS = ForecastSettings(
    max_context=8192, max_horizon=1024,
    quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    default_horizon=96, default_quantiles=(0.1, 0.5, 0.9),
)
COMPILE_SCOPE_ENV = 'VTSFM_TIMESFM_COMPILE_SCOPE'
COMPILE_SCOPES = ('core', 'wide', 'whole', 'positioned-eager', 'positioned')
FLIP_LAYOUT_ENV = 'VTSFM_TIMESFM_FLIP_LAYOUT'


def compile_scope(environment, execution, precision, *, continuous=False):
    """Select the default guarded core or an experimental wider FP32 scope."""
    scope = environment.get(COMPILE_SCOPE_ENV, 'core')
    if scope not in COMPILE_SCOPES:
        raise ValueError(f'{COMPILE_SCOPE_ENV} must be one of {COMPILE_SCOPES}, got {scope!r}')
    required_mode = 'eager' if scope == 'positioned-eager' else 'compile'
    if scope != 'core' and (execution != required_mode or precision not in (None, 'fp32') or continuous):
        raise ValueError(f'TimesFM {scope} requires whole-forecast serving with '
                         f'--forecast-execution {required_mode} and FP32')
    return scope


def flip_layout(environment, precision, *, independent_whole_forecast):
    """Select the experimental upstream flip geometry for independent FP32 requests."""
    layout = environment.get(FLIP_LAYOUT_ENV, 'packed')
    if layout not in ('packed', 'split'):
        raise ValueError(f"{FLIP_LAYOUT_ENV} must be 'packed' or 'split', got {layout!r}")
    if layout == 'split' and (precision not in (None, 'fp32') or not independent_whole_forecast):
        raise ValueError('TimesFM split flips require independent whole-forecast serving in FP32')
    return layout


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
    admission_limits=DEFAULT_LIMITS,
    admission_cost='vtsfm.adapters.timesfm25.admission:forecast_cost',
    execution_capabilities=ExecutionCapabilities(
        max_context=FORECAST_SETTINGS.max_context, max_horizon=FORECAST_SETTINGS.max_horizon,
        quantiles=FORECAST_SETTINGS.quantiles, input_patch_size=32, output_block_size=128,
        supports_point_forecast=False,
        supports_block_stepping=True,
        execution_options=(('normalize_inputs', 'true'), ('use_continuous_quantile_head', 'true'),
            ('force_flip_invariance', 'true'), ('infer_is_positive', 'true'),
            ('fix_quantile_crossing', 'true')),
    ),
)
