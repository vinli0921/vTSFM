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
    candidate = BatchedConformalIntervals(ContextForecaster(), batch_predict=callback, batch_size=7,
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


@pytest.mark.parametrize('batch_size', [1, np.int64(1)])
def test_float_initial_window_and_batch_size_one_keep_all_origins(batch_size):
    y = pd.Series(np.arange(30, dtype=float), name='sensor')
    callback = RecordingBatch()
    candidate = BatchedConformalIntervals(ContextForecaster(), batch_predict=callback,
                                         batch_size=batch_size, initial_window=0.2)
    original = ConformalIntervals(ContextForecaster(), initial_window=0.2)
    candidate.fit(y, fh=[1, 2])
    original.fit(y, fh=[1, 2])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    assert len(callback.calls) == 24


@pytest.mark.parametrize('failure', ['missing', 'scalar', 'count', 'index', 'nan', 'dtype'])
def test_bad_batch_results_use_native_residuals(failure):
    def callback(forecasters, targets):
        if failure == 'missing':
            return None
        if failure == 'scalar':
            return 0
        values = list(RecordingBatch()(forecasters, targets))
        if failure == 'count':
            return values[:-1]
        if failure == 'index':
            values[0] = values[0].iloc[::-1]
        elif failure == 'nan':
            values[0].iloc[0] = float('nan')
        else:
            values[0] = pd.Series('invalid', index=targets[0])
        return values
    y = pd.Series(np.arange(20, dtype=float))
    original = ConformalIntervals(ContextForecaster()).fit(y, fh=[1])
    candidate = BatchedConformalIntervals(ContextForecaster(), batch_predict=callback,
                                         batch_size=4).fit(y, fh=[1])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    pd.testing.assert_frame_equal(candidate.predict_interval(), original.predict_interval())
    assert len(candidate.forecasters_) == len(original.forecasters_)


@pytest.mark.parametrize('observed, forecast, dtype', [
    (2**53 + 1, 2**53, 'int64'),
    (2**53, 2**53 + 1, 'uint64'),
    (2**63 - 1, -(2**63), 'int64'),
])
def test_integer_residuals_are_subtracted_before_casting(observed, forecast, dtype):
    y = pd.Series(observed, index=pd.RangeIndex(12), dtype=dtype)

    def callback(forecasters, targets):
        return [pd.Series(forecast, index=target, dtype=dtype) for target in targets]

    candidate = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
        batch_predict=callback, batch_size=4).fit(y, fh=[1])
    assert candidate.residuals_matrix_.loc[5, 5] == float(observed - forecast)


@pytest.mark.parametrize('lazy', [False, True])
def test_batch_callback_type_errors_propagate(lazy):
    def broken_forecasts():
        yield None
        raise TypeError('callback failed')

    def callback(forecasters, targets):
        if lazy:
            return broken_forecasts()
        raise TypeError('callback failed')

    candidate = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
        batch_predict=callback, batch_size=4)
    with pytest.raises(TypeError, match='callback failed'):
        candidate.fit(pd.Series(np.arange(12, dtype=float)), fh=[1])


def test_missing_values_keep_native_residuals_and_intervals():
    class MissingValuesForecaster(ContextForecaster):
        _tags = {'capability:missing_values': True}

        def _predict(self, fh, X=None):
            result = super()._predict(fh, X)
            if len(self.prefix) % 3 == 0:
                result.iloc[0] = np.nan
            return result

    y = pd.Series(np.arange(20, dtype=float))
    y.iloc[8] = np.nan
    original = ConformalIntervals(MissingValuesForecaster(), initial_window=5).fit(y, fh=[1, 2])
    candidate = BatchedConformalIntervals(MissingValuesForecaster(), initial_window=5,
        batch_predict=RecordingBatch(), batch_size=4).fit(y, fh=[1, 2])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    pd.testing.assert_frame_equal(candidate.predict_interval(), original.predict_interval())
    assert pd.isna(candidate.residuals_matrix_.loc[6, 6])


def test_explicit_n_jobs_uses_native_calibration():
    y = pd.Series(np.arange(20, dtype=float))
    original = ConformalIntervals(ContextForecaster(), n_jobs=1).fit(y, fh=[1])
    callback = RecordingBatch()
    candidate = BatchedConformalIntervals(ContextForecaster(), n_jobs=1,
                                         batch_predict=callback).fit(y, fh=[1])
    assert callback.calls == []
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)


@pytest.mark.parametrize('size', [True, 0, -1, 257, 1.5])
def test_invalid_batch_limits_are_rejected(size):
    with pytest.raises(ValueError, match='batch_size'):
        BatchedConformalIntervals(ContextForecaster(), batch_predict=RecordingBatch(), batch_size=size)


def test_original_constructor_order_no_callback_and_clone_are_preserved():
    candidate = BatchedConformalIntervals(ContextForecaster(), 'empirical', 5,
                                         None, False, 1)
    assert candidate.batch_predict is None
    cloned = candidate.clone()
    assert cloned.get_params()['batch_predict'] is None
    y = pd.Series(np.arange(24, dtype=float))
    original = ConformalIntervals(ContextForecaster(), 'empirical', 5, None, False, 1)
    candidate.fit(y, fh=[1, 2, 3])
    original.fit(y, fh=[1, 2, 3])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    pd.testing.assert_frame_equal(candidate.predict_interval(), original.predict_interval())
    assert candidate.get_tag('capability:update') == original.get_tag('capability:update')


def test_callback_clone_and_deferred_forecasting_horizon_remain_usable():
    candidate = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                         batch_predict=RecordingBatch(), batch_size=7).clone()
    assert isinstance(candidate.batch_predict, RecordingBatch)
    assert candidate.batch_size == 7
    original = ConformalIntervals(ContextForecaster(), initial_window=5)
    y = pd.Series(np.arange(24, dtype=float))
    candidate.fit(y)
    original.fit(y)
    pd.testing.assert_frame_equal(candidate.predict_interval(fh=[1, 2, 3]),
                                  original.predict_interval(fh=[1, 2, 3]))
    assert candidate.residuals_matrix_ is original.residuals_matrix_ is None


@pytest.mark.parametrize('with_callback', [False, True])
def test_origin_sampling_uses_original_algorithm(with_callback):
    callback = RecordingBatch() if with_callback else None
    candidate = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                         sample_frac=0.5, batch_predict=callback)
    original = ConformalIntervals(ContextForecaster(), initial_window=5, sample_frac=0.5)
    y = pd.Series(np.arange(30, dtype=float))
    np.random.seed(42)
    original.fit(y, fh=[1, 2])
    np.random.seed(42)
    candidate.fit(y, fh=[1, 2])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    assert callback is None or callback.calls == []


def test_covariates_delegate_to_the_original_residual_loop():
    class CovariateForecaster(ContextForecaster):
        _tags = {'capability:exogenous': True}

        def _predict(self, fh, X=None):
            assert X is not None
            return super()._predict(fh, X) + X.iloc[:len(fh), 0].to_numpy()

    callback = RecordingBatch()
    y = pd.Series(np.arange(30, dtype=float))
    X = pd.DataFrame({'signal': np.sin(np.arange(33))})
    original = ConformalIntervals(CovariateForecaster(), initial_window=5)
    candidate = BatchedConformalIntervals(CovariateForecaster(), initial_window=5,
                                         batch_predict=callback)
    original.fit(y, X=X.iloc[:30], fh=[1, 2, 3])
    candidate.fit(y, X=X.iloc[:30], fh=[1, 2, 3])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    pd.testing.assert_frame_equal(candidate.predict_interval(X=X.iloc[30:]),
                                  original.predict_interval(X=X.iloc[30:]))
    assert callback.calls == []


@pytest.mark.parametrize('update_params', [False, True])
def test_native_updates_preserve_residuals_and_capabilities(update_params):
    callback = RecordingBatch()
    y = pd.Series(np.arange(28, dtype=float))
    original = ConformalIntervals(ContextForecaster(), initial_window=5)
    candidate = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                         batch_predict=callback)
    original.fit(y.iloc[:24], fh=[1, 2])
    candidate.fit(y.iloc[:24], fh=[1, 2])
    callback.calls.clear()
    # Full observed history is a supported native update input. sktime's update
    # loop extends retained per-origin forecasts rather than rebatching them.
    original.update(y, update_params=update_params)
    candidate.update(y, update_params=update_params)
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
    pd.testing.assert_frame_equal(candidate.predict_interval(), original.predict_interval())
    assert callback.calls == []


def test_toto_callback_keeps_other_forecasters_serial_without_torch():
    batcher = pytest.importorskip('vtsfm.recipes.toto').TotoCalibrationBatcher()
    y = pd.Series(np.arange(20, dtype=float), name='sensor')
    forecasters = [ContextForecaster().fit(y.iloc[:origin]) for origin in [10, 11]]
    targets = [y.iloc[origin:].index for origin in [10, 11]]
    expected = RecordingBatch()(forecasters, targets)
    actual = batcher(forecasters, targets)
    for left, right in zip(actual, expected):
        pd.testing.assert_series_equal(left, right)


class RandomFitForecaster(BaseForecaster):
    """Fit/predict ordering matters even when predictions are called serially."""

    _tags = {'y_inner_mtype': 'pd.Series', 'requires-fh-in-fit': False}

    def __init__(self, seed=None):
        self.seed = seed
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.draw = np.random.random()
        return self

    def _predict(self, fh, X=None):
        return pd.Series(self.draw + np.random.random(len(fh)),
                         index=fh.to_absolute(self.cutoff).to_pandas())


@pytest.mark.parametrize('known_type', [False, True])
def test_native_fallback_preserves_fit_predict_rng_order(known_type):
    batcher = pytest.importorskip('vtsfm.recipes.toto').TotoCalibrationBatcher()
    forecaster = RandomFitForecaster()
    if known_type:
        # An unseeded Toto forecaster also uses the original loop.
        cls = type('Toto2Forecaster', (RandomFitForecaster,), {'__module__': 'sktime.forecasting.toto2'})
        forecaster = cls()
    original = ConformalIntervals(forecaster, initial_window=5)
    candidate = BatchedConformalIntervals(forecaster, initial_window=5,
                                         batch_predict=batcher)
    y = pd.Series(np.arange(18, dtype=float))
    np.random.seed(42)
    original.fit(y, fh=[1, 2])
    np.random.seed(42)
    candidate.fit(y, fh=[1, 2])
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_)
