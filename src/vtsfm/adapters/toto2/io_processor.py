"""Toto numerical request/response handling."""

from vtsfm.adapters.pooling import ForecastIOProcessor
from .spec import FORECAST_SETTINGS, TOTO2
from .admission import forecast_cost


class TotoIOProcessor(ForecastIOProcessor):
    settings = FORECAST_SETTINGS
    admission_spec = TOTO2
    admission_cost = staticmethod(forecast_cost)
