"""TimesFM numerical request/response handling."""

from vtsfm.adapters.pooling import ForecastIOProcessor
from .spec import FORECAST_SETTINGS, TIMESFM25
from .admission import forecast_cost


class TimesFMIOProcessor(ForecastIOProcessor):
    settings = FORECAST_SETTINGS
    admission_spec = TIMESFM25
    admission_cost = staticmethod(forecast_cost)
