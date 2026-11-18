"""Whole-forecast Toto pooling adapter."""

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
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
