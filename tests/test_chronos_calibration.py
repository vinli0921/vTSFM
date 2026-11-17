"""Native callback boundaries: patch buckets, original fallbacks and routing."""

from contextlib import nullcontext
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
ForecastingHorizon = pytest.importorskip('sktime.forecasting.base').ForecastingHorizon
ChronosCalibrationBatcher = pytest.importorskip('vtsfm.recipes.chronos').ChronosCalibrationBatcher


class Pipeline:
    quantiles = [0.1, 0.5, 0.9]
    model_context_length = 16
    model_output_patch_size = 4
    model_prediction_length = 8

    def __init__(self):
        self.calls = []

    def predict(self, inputs, *, prediction_length, **options):
        self.calls.append((len(inputs), prediction_length, options))
        # Deliberately horizon sensitive across patch boundaries. Arbitrarily
        # requesting the largest horizon then cropping would change predictions.
        offset = math.ceil(prediction_length / self.model_output_patch_size)
        return [np.array([[np.asarray(item['target']).mean() + offset
                           + np.arange(1, prediction_length + 1) * 0.13 + delta
                           for delta in [-1, 0, 1]]]) for item in inputs]


def native_predict(self, fh):
    horizon = int(max(fh.to_relative(self.cutoff)))
    result, = self.model_pipeline.predict([{'target': self._context}],
        prediction_length=horizon, batch_size=256, context_length=16,
        cross_learning=self._config['cross_learning'], limit_prediction_length=False)
    relative = np.asarray(fh.to_relative(self.cutoff)) - 1
    return pd.Series(result[0, 1, relative], index=fh.to_absolute(self.cutoff).to_pandas(), name='sensor')


# Lightweight API doubles; the optional native suite below exercises the actual
# sktime forecaster and Chronos model without checkpoint downloads.
FittedChronos = type('Chronos2Forecaster', (), {
    '__module__': 'sktime.forecasting.chronos2', 'predict': native_predict,
    '_get_varnames': lambda self: ['sensor']})


def requests(pipeline, horizons=(4, 3, 2, 1), widths=None):
    widths = widths or [16] * len(horizons)
    forecasters, targets = [], []
    for number, (width, horizon) in enumerate(zip(widths, horizons)):
        start = 20 + number
        fc = FittedChronos()
        fc._context = np.arange(width, dtype=float)[None, :] + number
        fc._cur_X = None
        fc.cutoff = pd.Index([start - 1])
        fc.model_pipeline = pipeline
        fc._config = {'context_length': None, 'cross_learning': False,
                      'limit_prediction_length': False}
        fc._seed = 42
        fc.seed = 42
        forecasters.append(fc)
        targets.append(pd.RangeIndex(start, start + horizon))
    return tuple(forecasters), tuple(targets)


@pytest.fixture
def seeds(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'vtsfm.recipes._chronos_ops',
                        SimpleNamespace(ChronosOriginMode=lambda *args: nullcontext()))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(set_seed=calls.append))
    return calls


def test_matching_native_patch_counts_batch_varying_horizons(seeds):
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline)
    expected = [fc.predict(ForecastingHorizon(y, is_relative=False))
                for fc, y in zip(forecasters, targets)]
    pipeline.calls.clear()
    result = ChronosCalibrationBatcher()(forecasters, targets)
    assert [call[:2] for call in pipeline.calls] == [(4, 4)]
    assert pipeline.calls[0][2]['cross_learning'] is False
    assert seeds == [42]
    for actual, wanted in zip(result, expected):
        pd.testing.assert_series_equal(actual, wanted)


@pytest.mark.parametrize('malformed', ['count', 'shape'])
def test_invalid_batched_chronos_output_retries_native_origins(seeds, malformed):
    class BadBatchPipeline(Pipeline):
        def predict(self, inputs, **options):
            outputs = super().predict(inputs, **options)
            if len(inputs) > 1:
                if malformed == 'count':
                    return outputs[:-1]
                outputs[0] = outputs[0][..., :-1]
            return outputs

    pipeline = BadBatchPipeline()
    forecasters, targets = requests(pipeline)
    wanted = [fc.predict(ForecastingHorizon(index, is_relative=False))
              for fc, index in zip(forecasters, targets)]
    pipeline.calls.clear()
    actual = ChronosCalibrationBatcher()(forecasters, targets)
    assert [call[0] for call in pipeline.calls] == ([4, 1, 1, 1, 1] if malformed == 'count' else [4, 1])
    for prediction, expected in zip(actual, wanted):
        pd.testing.assert_series_equal(prediction, expected)


def test_patch_boundaries_and_long_horizons_preserve_native_schedule(seeds):
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline, horizons=[4, 5, 3, 6, 9])
    expected = [fc.predict(ForecastingHorizon(y, is_relative=False))
                for fc, y in zip(forecasters, targets)]
    pipeline.calls.clear()
    result = ChronosCalibrationBatcher()(forecasters, targets)
    assert [call[:2] for call in pipeline.calls] == [(2, 4), (2, 6), (1, 9)]
    for actual, wanted in zip(result, expected):
        pd.testing.assert_series_equal(actual, wanted)


def test_distinct_context_widths_are_not_silently_padded():
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline, widths=[12, 13, 14, 15])
    result = ChronosCalibrationBatcher()(forecasters, targets)
    assert [call[:2] for call in pipeline.calls] == [(1, 4), (1, 3), (1, 2), (1, 1)]
    assert len(result) == 4


def test_point_budget_splits_partial_batches_and_preserves_oversized_serial_calls(seeds):
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline, horizons=[4, 3, 2, 1, 1])
    outputs = ChronosCalibrationBatcher(max_batch_points=56)(forecasters, targets)
    assert [call[:2] for call in pipeline.calls] == [(2, 4), (2, 2), (1, 1)]
    assert len(outputs) == 5
    pipeline.calls.clear()
    ChronosCalibrationBatcher(max_batch_points=1)(forecasters, targets)
    assert all(rows == 1 for rows, _, _ in pipeline.calls)


@pytest.mark.parametrize('mode', ['cross_learning', 'covariates', 'other_model', 'seed', 'sparse_fh'])
def test_unsupported_batching_keeps_native_serial_forecasts(mode, seeds):
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline, horizons=[4, 3])
    if mode == 'cross_learning':
        for fc in forecasters:
            fc._config['cross_learning'] = True
    elif mode == 'covariates':
        for fc in forecasters:
            fc._cur_X = pd.DataFrame({'signal': [1]})
    elif mode == 'other_model':
        class OtherModel:
            predict = native_predict
        for fc in forecasters:
            fc.__class__ = OtherModel
    elif mode == 'seed':
        forecasters[1]._seed = 13
    else:
        targets = tuple(target[::2] for target in targets)
    result = ChronosCalibrationBatcher()(forecasters, targets)
    assert len(result) == 2 and all(call[0] == 1 for call in pipeline.calls)
