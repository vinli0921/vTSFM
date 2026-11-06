"""Whole-forecast Toto pooling adapter."""

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
from .runtime import TotoRuntime
from .spec import FORECAST_SETTINGS


@MULTIMODAL_REGISTRY.register_processor(
    ForecastMultiModalProcessor, info=ForecastProcessingInfo, dummy_inputs=ForecastInputBuilder,
)
class VTSFMTotoForForecasting(ForecastModel):
    runtime_class = TotoRuntime
    settings = FORECAST_SETTINGS
