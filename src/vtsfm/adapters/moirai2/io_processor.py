"""Moirai numerical request/response handling."""

from vtsfm.adapters.pooling import ForecastIOProcessor
from .spec import FORECAST_SETTINGS, MOIRAI2
from .admission import forecast_cost


class MoiraiIOProcessor(ForecastIOProcessor):
    settings = FORECAST_SETTINGS
    admission_spec = MOIRAI2
    admission_cost = staticmethod(forecast_cost)
