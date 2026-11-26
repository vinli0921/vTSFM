"""Whole-forecast Toto pooling adapter."""

import os

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
from vtsfm.compilation import execution_mode
from vtsfm.precision import PRECISION_ENV, TF32_OVERRIDE_ENV

from .graph_config import (
    FAST_SCALING_ENV, GRAPH_SCALING_MODE, GRAPH_SHAPE_ENV,
    parse_graph_shape, validate_graph_execution, validate_graph_shape,
)
from .runtime import TotoRuntime
from .spec import FORECAST_SETTINGS, TOTO2
from .admission import forecast_cost


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo, dummy_inputs=ForecastInputBuilder,
)
class VTSFMTotoForForecasting(ForecastModel):
    runtime_class = TotoRuntime
    settings = FORECAST_SETTINGS
    batch_spec = TOTO2
    admission_cost = staticmethod(forecast_cost)

    def __init__(self, *, vllm_config, prefix=''):
        shape = parse_graph_shape(os.environ.get(GRAPH_SHAPE_ENV))
        if shape is not None:
            validate_graph_execution(os.environ.get(PRECISION_ENV), execution_mode())
            model_config = vllm_config.model_config
            if (os.environ.get(PRECISION_ENV) != 'fp32'
                    or os.environ.get(TF32_OVERRIDE_ENV) != '0'
                    or os.environ.get(FAST_SCALING_ENV) != GRAPH_SCALING_MODE):
                raise ValueError('Toto CUDA graphs require FP32, TF32 disabled and stride-aware scaling')
            if (not model_config.enforce_eager or model_config.enable_sleep_mode
                    or vllm_config.scheduler_config.async_scheduling or vllm_config.use_v2_model_runner
                    or vllm_config.parallel_config.data_parallel_size != 1
                    or vllm_config.lora_config is not None or vllm_config.weight_transfer_config is not None):
                raise ValueError('Toto CUDA graphs require synchronous V1 pooling with fixed weights and no sleep mode')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if shape is not None:
            from .graph_serving import TotoForecastGraph

            validate_graph_shape(shape, self.forecast_limits)
            self.runtime.fixed_forecast = TotoForecastGraph(shape)

    def warm_fixed_forecast(self):
        if self.runtime.fixed_forecast is not None:
            self.runtime.fixed_forecast.warm(self.runtime)

    def close_fixed_forecast(self):
        if self.runtime.fixed_forecast is not None:
            self.runtime.fixed_forecast.close()

    def fixed_forecast_snapshot(self):
        owner = self.runtime.fixed_forecast
        if owner is None:
            return {'mode': 'eager', 'shape': None, 'setup': None,
                    'capture_attempts': 0, 'graph_replays': 0, 'fallbacks': 0}
        return owner.snapshot()
