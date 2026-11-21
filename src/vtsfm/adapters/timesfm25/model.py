"""Whole-forecast TimesFM pooling adapter."""

import os

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
from .runtime import TimesFMRuntime
from .spec import FORECAST_SETTINGS, TIMESFM25
from .admission import forecast_cost
from .graph_config import FORECAST_MODE_ENV, FORECAST_MODES, FORECAST_GRAPH_SHAPE
from vtsfm.stepping import CONTINUOUS_ENV, validate_block_configuration


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo, dummy_inputs=ForecastInputBuilder,
)
class VTSFMTimesFMForForecasting(ForecastModel):
    runtime_class = TimesFMRuntime
    settings = FORECAST_SETTINGS
    batch_spec = TIMESFM25
    admission_cost = staticmethod(forecast_cost)

    def __init__(self, *, vllm_config, prefix=''):
        continuous = os.environ.get(CONTINUOUS_ENV) == '1'
        mode = os.environ.get(FORECAST_MODE_ENV)
        if continuous:
            validate_block_configuration(vllm_config)
        if mode is not None:
            scheduler = vllm_config.scheduler_config
            model_config = vllm_config.model_config
            if (mode not in FORECAST_MODES or continuous or scheduler.max_num_seqs != 1
                    or scheduler.async_scheduling or not model_config.enforce_eager
                    or model_config.enable_sleep_mode or vllm_config.use_v2_model_runner
                    or vllm_config.parallel_config.data_parallel_size != 1
                    or vllm_config.lora_config is not None
                    or vllm_config.weight_transfer_config is not None):
                raise ValueError('TimesFM forecast mode requires a known mode and synchronous '
                                 'single-slot V1 pooling with eager runner, fixed weights and no sleep mode')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if mode is not None:
            if self.forecast_batch_rows:
                raise ValueError('TimesFM forecast mode requires whole forecasts in one slot')
            from .graph_serving import ServingFixedForecast

            self.forecast_limits.validate_request(forecast_cost(*FORECAST_GRAPH_SHAPE))
            self.runtime.fixed_forecast = ServingFixedForecast(mode)
        if continuous:
            from vtsfm.block_pooling import BlockExecutor, BlockForecastPooler
            from .stepping import TimesFMStepper

            self.stepper = TimesFMStepper(self.runtime)
            self.block_executor = BlockExecutor(self.stepper, self.forecast_limits,
                forecast_cost, self.settings.quantiles, self.forecast_batch_rows)
            self.pooler = BlockForecastPooler(self.block_executor)

    def warm_fixed_forecast(self):
        if self.runtime.fixed_forecast is not None:
            self.runtime.fixed_forecast.warm(self.runtime)

    def close_fixed_forecast(self):
        if self.runtime.fixed_forecast is not None:
            self.runtime.fixed_forecast.close()

    def fixed_forecast_snapshot(self):
        policy = self.runtime.fixed_forecast
        if policy is None:
            return {'mode': 'default', 'calls': 0, 'fallbacks': 0, 'graph_replays': 0,
                    'capture_attempts': 0, 'setup': None, 'fixed_pool_bytes': 0, 'leased_cache_bytes': 0}
        return policy.snapshot()
