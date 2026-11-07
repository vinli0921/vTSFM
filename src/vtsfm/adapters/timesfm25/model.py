"""Whole-forecast TimesFM pooling adapter."""

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
from .runtime import TimesFMRuntime
from .spec import FORECAST_SETTINGS, TIMESFM25
from .admission import forecast_cost


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo, dummy_inputs=ForecastInputBuilder,
)
class VTSFMTimesFMForForecasting(ForecastModel):
    runtime_class = TimesFMRuntime
    settings = FORECAST_SETTINGS
    batch_spec = TIMESFM25
    admission_cost = staticmethod(forecast_cost)
