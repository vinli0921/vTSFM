"""The batched origin loop must retain sktime's residuals and interval semantics."""

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
base = pytest.importorskip('sktime.forecasting.base')
BaseForecaster = base.BaseForecaster
ForecastingHorizon = base.ForecastingHorizon
ConformalIntervals = pytest.importorskip('sktime.forecasting.conformal').ConformalIntervals
BatchedConformalIntervals = pytest.importorskip('vtsfm.recipes.conformal').BatchedConformalIntervals


class ContextForecaster(BaseForecaster):
    """Deterministic point forecaster whose output depends on the entire prefix."""

    _tags = {'y_inner_mtype': 'pd.Series', 'capability:exogenous': False,
             'requires-fh-in-fit': False}

    def _fit(self, y, X=None, fh=None):
        self.prefix = y.copy()
        return self

    def _predict(self, fh, X=None):
        steps = np.asarray(fh.to_relative(self.cutoff))
        values = self.prefix.mean() + self.prefix.iloc[-1] / 7 + steps * 0.13
        return pd.Series(values, index=fh.to_absolute(self.cutoff).to_pandas(),
                         name=self.prefix.name)


class RecordingBatch:
    def __init__(self):
        self.calls = []

    def __call__(self, forecasters, targets):
        assert all(isinstance(target, pd.Index) for target in targets)
        self.calls.append(tuple((len(fc.prefix), tuple(target))
                                for fc, target in zip(forecasters, targets)))
        return tuple(fc.predict(fh=ForecastingHorizon(target, is_relative=False))
                     for fc, target in zip(forecasters, targets))


@pytest.mark.parametrize('index', [
    pd.RangeIndex(37, name='time'),
    pd.date_range('2020-01-01', periods=37, freq='h', name='time'),
    pd.period_range('2020-01', periods=37, freq='M', name='time'),
])
@pytest.mark.parametrize('method', ConformalIntervals.ALLOWED_METHODS)
def test_residuals_and_intervals_match_original_origin_loop(index, method):
    y = pd.Series(np.sin(np.arange(37) / 3) + np.arange(37) / 10,
                  index=index, name='sensor')
    original = ConformalIntervals(ContextForecaster(), initial_window=5, method=method)
    callback = RecordingBatch()
    candidate = BatchedConformalIntervals(ContextForecaster(), callback, batch_size=7,
                                         initial_window=5, method=method)
    original.fit(y, fh=[1, 2, 3, 5])
    candidate.fit(y, fh=[1, 2, 3, 5])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    pd.testing.assert_frame_equal(candidate.predict_interval(coverage=[0.5, 0.9]),
                                  original.predict_interval(coverage=[0.5, 0.9]))
    assert [len(call) for call in callback.calls] == [7, 7, 7, 7, 4]
    positions = [length for call in callback.calls for length, _ in call]
    assert positions == list(range(5, 37))
    assert [row['id'] for row in candidate.forecasters_] == [str(i) for i in index[5:]]


def test_float_initial_window_and_batch_size_one_keep_all_origins():
    y = pd.Series(np.arange(30, dtype=float), name='sensor')
    callback = RecordingBatch()
    candidate = BatchedConformalIntervals(ContextForecaster(), callback,
                                         batch_size=1, initial_window=0.2)
    original = ConformalIntervals(ContextForecaster(), initial_window=0.2)
    candidate.fit(y, fh=[1, 2])
    original.fit(y, fh=[1, 2])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    assert len(callback.calls) == 24


@pytest.mark.parametrize('failure', ['count', 'index', 'nan'])
def test_bad_batch_results_fail_instead_of_changing_calibration(failure):
    def callback(forecasters, targets):
        values = list(RecordingBatch()(forecasters, targets))
        if failure == 'count':
            return values[:-1]
        if failure == 'index':
            values[0] = values[0].iloc[::-1]
        else:
            values[0].iloc[0] = float('nan')
        return values
    candidate = BatchedConformalIntervals(ContextForecaster(), callback, batch_size=4)
    with pytest.raises(ValueError, match='Calibration'):
        candidate.fit(pd.Series(np.arange(20, dtype=float)), fh=[1])


@pytest.mark.parametrize('size', [True, 0, -1, 257, 1.5])
def test_invalid_batch_limits_are_rejected(size):
    with pytest.raises(ValueError, match='batch_size'):
        BatchedConformalIntervals(ContextForecaster(), RecordingBatch(), batch_size=size)


def test_sampled_origins_and_covariates_are_explicitly_unsupported():
    with pytest.raises(ValueError, match='sampled'):
        BatchedConformalIntervals(ContextForecaster(), RecordingBatch(), sample_frac=0.5)
    candidate = BatchedConformalIntervals(ContextForecaster(), RecordingBatch())
    with pytest.raises(ValueError, match='without X'):
        candidate._compute_sliding_residuals(pd.Series(range(20)), pd.DataFrame({'x': range(20)}),
                                             ContextForecaster(), 4, None)


def test_updates_fail_explicitly_instead_of_using_a_different_calibration_loop():
    candidate = BatchedConformalIntervals(ContextForecaster(), RecordingBatch())
    candidate.fit(pd.Series(np.arange(20, dtype=float)), fh=[1])
    with pytest.raises(NotImplementedError, match='updates'):
        candidate.update(pd.Series([20., 21., 22.], index=[20, 21, 22]))
