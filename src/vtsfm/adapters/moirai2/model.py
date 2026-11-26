"""Whole-forecast Moirai pooling adapter."""

import os

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
from vtsfm.stepping import CONTINUOUS_ENV, validate_block_configuration
from .runtime import MoiraiRuntime
from .spec import FORECAST_SETTINGS, MOIRAI2
from .admission import forecast_cost


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo, dummy_inputs=ForecastInputBuilder,
)
class VTSFMMoiraiForForecasting(ForecastModel):
    runtime_class = MoiraiRuntime
    settings = FORECAST_SETTINGS
    batch_spec = MOIRAI2
    admission_cost = staticmethod(forecast_cost)

    def __init__(self, *, vllm_config, prefix=''):
        continuous = os.environ.get(CONTINUOUS_ENV) == '1'
        if continuous:
            validate_block_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if continuous:
            from vtsfm.block_pooling import BlockExecutor, BlockForecastPooler
            from .stepping import MoiraiStepper

            self.stepper = MoiraiStepper(self.runtime)
            self.block_executor = BlockExecutor(self.stepper, self.forecast_limits,
                forecast_cost, self.settings.quantiles, self.forecast_batch_rows)
            self.pooler = BlockForecastPooler(self.block_executor)
