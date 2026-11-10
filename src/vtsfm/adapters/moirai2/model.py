"""Whole-forecast Moirai pooling adapter."""

from vllm.multimodal import MULTIMODAL_REGISTRY
from vtsfm.adapters.multimodal import ForecastInputBuilder, ForecastMultiModalProcessor, ForecastProcessingInfo
from vtsfm.adapters.pooling import ForecastModel
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
