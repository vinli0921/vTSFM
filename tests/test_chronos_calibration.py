"""CPU boundary tests for native calibration calls; no model weights are loaded."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
ForecastingHorizon = pytest.importorskip('sktime.forecasting.base').ForecastingHorizon
ChronosCalibrationBatcher = pytest.importorskip('vtsfm.recipes.chronos').ChronosCalibrationBatcher


class Pipeline:
    quantiles = [0.1, 0.5, 0.9]
    model_context_length = 8192

    def __init__(self, horizon_dependent=False):
        self.calls = []
        self.horizon_dependent = horizon_dependent

    def predict(self, inputs, *, prediction_length, **options):
        self.calls.append((len(inputs), prediction_length, options))
        offset = prediction_length if self.horizon_dependent else 0
        return [np.array([[np.asarray(item['target']).mean() + offset
                           + np.arange(1, prediction_length + 1) * 0.13 + delta
                           for delta in [-1, 0, 1]]]) for item in inputs]


def native_predict(self, fh):
    horizon = int(max(fh.to_relative(self.cutoff)))
    result, = self.model_pipeline.predict([{'target': self._context}],
        prediction_length=horizon, batch_size=256, context_length=8192,
        cross_learning=False, limit_prediction_length=False)
    return pd.Series(result[0, 1], index=fh.to_absolute(self.cutoff).to_pandas(), name='sensor')


# Stand-ins for the documented fitted sktime attributes. Real-model parity is
# a separate opt-in check; the test pipeline deliberately exposes horizon drift.
FittedChronos = type('Chronos2Forecaster', (), {
    '__module__': 'sktime.forecasting.chronos2', 'predict': native_predict,
    '_get_varnames': lambda self: ['sensor']})


def requests(pipeline):
    forecasters, targets = [], []
    for width in [12, 13, 14, 15]:
        target = pd.RangeIndex(width, 16)
        fc = FittedChronos()
        fc._context = np.arange(width, dtype=float)[None, :]
        fc._cur_X = None
        fc.cutoff = pd.Index([width - 1])
        fc.model_pipeline = pipeline
        fc._config = {'context_length': None, 'cross_learning': False,
                      'limit_prediction_length': False}
        fc._seed = 42
        forecasters.append(fc)
        targets.append(target)
    return tuple(forecasters), tuple(targets)


def test_qualified_padding_batches_calls_and_restores_original_order(monkeypatch):
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline)
    expected = [fc.predict(ForecastingHorizon(y, is_relative=False))
                for fc, y in zip(forecasters, targets)]
    pipeline.calls.clear()
    seeds = []
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(set_seed=seeds.append))
    result = ChronosCalibrationBatcher(context_padding_invariant=True,
                                      horizon_prefix_invariant=True)(forecasters, targets)
    assert len(pipeline.calls) == 1 and pipeline.calls[0][:2] == (4, 4)
    assert pipeline.calls[0][2]['cross_learning'] is False
    assert seeds == [42]
    for actual, wanted in zip(result, expected):
        pd.testing.assert_series_equal(actual, wanted)


def test_default_does_not_pad_a_horizon_sensitive_model():
    pipeline = Pipeline(horizon_dependent=True)
    forecasters, targets = requests(pipeline)
    result = ChronosCalibrationBatcher()(forecasters, targets)
    assert [(rows, horizon) for rows, horizon, _ in pipeline.calls] == [(1, 4), (1, 3), (1, 2), (1, 1)]
    assert result[-1].iloc[0] == pytest.approx(forecasters[-1]._context.mean() + 1 + 0.13)


def test_point_budget_splits_batches_without_dropping_the_tail(monkeypatch):
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(set_seed=lambda seed: None))
    outputs = ChronosCalibrationBatcher(True, True, max_batch_points=55)(forecasters, targets)
    assert [(rows, horizon) for rows, horizon, _ in pipeline.calls] == [(2, 4), (2, 2)]
    assert len(outputs) == 4 and [len(y) for y in outputs] == [4, 3, 2, 1]


def test_oversized_origin_and_cross_learning_are_rejected_before_inference():
    pipeline = Pipeline()
    forecasters, targets = requests(pipeline)
    with pytest.raises(ValueError, match='max_batch_points'):
        ChronosCalibrationBatcher(max_batch_points=15)(forecasters, targets)
    assert not pipeline.calls
    forecasters[0]._config['cross_learning'] = True
    with pytest.raises(ValueError, match='cross_learning'):
        ChronosCalibrationBatcher()(forecasters, targets)
    assert not pipeline.calls
