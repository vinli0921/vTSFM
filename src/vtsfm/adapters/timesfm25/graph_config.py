"""CPU-safe settings for the experimental fixed-shape whole-forecast path."""

FORECAST_MODE_ENV = 'VTSFM_TIMESFM_FORECAST_MODE'
FORECAST_MODES = ('dense-graph', 'paged-graph', 'dense-eager', 'paged-eager')
# Rows, context length, horizon; the measured numerical envelope is deliberately narrow.
FORECAST_GRAPH_SHAPE = (2, 512, 1024)
