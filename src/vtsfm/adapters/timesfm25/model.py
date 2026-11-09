"""Whole-forecast TimesFM pooling adapter."""

import os

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
from .runtime import TimesFMRuntime
from .spec import FORECAST_SETTINGS, TIMESFM25
from .admission import forecast_cost
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
        if os.environ.get(CONTINUOUS_ENV) == '1':
            validate_block_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if os.environ.get(CONTINUOUS_ENV) == '1':
            from vtsfm.block_pooling import BlockExecutor, BlockForecastPooler
            from .stepping import TimesFMStepper

            self.stepper = TimesFMStepper(self.runtime)
            self.block_executor = BlockExecutor(self.stepper, self.forecast_limits,
                forecast_cost, self.settings.quantiles, self.forecast_batch_rows)
            self.pooler = BlockForecastPooler(self.block_executor)
