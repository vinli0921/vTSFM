"""TimesFM numerical request/response handling."""

from vtsfm.adapters.pooling import ForecastIOProcessor
from .spec import FORECAST_SETTINGS


class TimesFMIOProcessor(ForecastIOProcessor):
    settings = FORECAST_SETTINGS
