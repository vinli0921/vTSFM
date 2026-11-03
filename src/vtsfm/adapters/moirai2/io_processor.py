"""Moirai numerical request/response handling."""

from vtsfm.adapters.pooling import ForecastIOProcessor
from .spec import FORECAST_SETTINGS


class MoiraiIOProcessor(ForecastIOProcessor):
    settings = FORECAST_SETTINGS
