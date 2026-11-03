"""Toto numerical request/response handling."""

from vtsfm.adapters.pooling import ForecastIOProcessor
from .spec import FORECAST_SETTINGS


class TotoIOProcessor(ForecastIOProcessor):
    settings = FORECAST_SETTINGS
