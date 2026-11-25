"""Capped calibration (#19) must keep every residual sktime's intervals read."""

import numpy as np
import pytest

from test_conformal_batching import (
    BatchedConformalIntervals, ConformalIntervals, ContextForecaster, RecordingBatch, pd,
)

INDEXES = [
    pd.RangeIndex(37, name='time'),
    pd.date_range('2020-01-01', periods=37, freq='h', name='time'),
    pd.period_range('2020-01', periods=37, freq='M', name='time'),
]
# Contiguous, gapped, and longer than the suffixes of the last calibration origins.
HORIZONS = [[1], [1, 2, 3], [1, 2, 3, 5], list(range(1, 13))]


class FitRecorder(ContextForecaster):
    """Logs (history length, horizon length) for every fit; clones share the log."""

    fits = []

    def _fit(self, y, X=None, fh=None):
        FitRecorder.fits.append((len(y), None if fh is None else len(fh)))
        return super()._fit(y, X, fh)


def _pair(y, fh, method='empirical', callback=None, **options):
    original = ConformalIntervals(ContextForecaster(), initial_window=5, method=method, **options)
    FitRecorder.fits.clear()
    capped = BatchedConformalIntervals(FitRecorder(), initial_window=5, method=method,
                                       batch_predict=callback, batch_size=7,
                                       cap_calibration_horizon=True, **options)
    return original.fit(y, fh=fh), capped.fit(y, fh=fh)


def _requested_lengths(series_length):
    """Horizon lengths of calibration-origin fits (the final fit uses the whole series)."""
    return [steps for history, steps in FitRecorder.fits if history < series_length]


@pytest.mark.parametrize('index', INDEXES)
@pytest.mark.parametrize('method', ConformalIntervals.ALLOWED_METHODS)
@pytest.mark.parametrize('fh', HORIZONS)
@pytest.mark.parametrize('with_callback', [False, True])
def test_capped_calibration_keeps_read_residuals_and_intervals(index, method, fh, with_callback):
    y = pd.Series(np.sin(np.arange(37) / 3) + np.arange(37) / 10, index=index, name='sensor')
    callback = RecordingBatch() if with_callback else None
    original, capped = _pair(y, fh, method, callback)
    lead = max(fh) + 1
    assert capped.calibration_horizon_ == lead
    # Every origin asks for the cap, or the shorter remaining suffix at the end.
    assert _requested_lengths(len(y)) == [min(lead, 37 - origin) for origin in range(5, 37)]
    if with_callback:
        assert max(len(target) for call in callback.calls for _, target in call) == lead
    full, short = original.residuals_matrix_.to_numpy(), capped.residuals_matrix_.to_numpy()
    for offset in fh:
        np.testing.assert_array_equal(np.diagonal(short, offset), np.diagonal(full, offset))
    # Nothing beyond the cap was forecast.
    assert np.isnan(np.diagonal(short, lead)).all()
    pd.testing.assert_frame_equal(capped.predict_interval(coverage=[0.5, 0.9]),
                                  original.predict_interval(coverage=[0.5, 0.9]))


def test_interval_for_step_h_reads_lead_h_plus_one():
    """Regression for the off-by-one: a cap of max(fh) would drop lead H + 1."""

    class ShortCap(BatchedConformalIntervals):
        def _needed_lead(self, fh):
            return super()._needed_lead(fh) - 1

    y = pd.Series(np.sin(np.arange(37) / 3) + np.arange(37) / 10, name='sensor')
    original, capped = _pair(y, [1, 2, 3])
    short = ShortCap(ContextForecaster(), initial_window=5, cap_calibration_horizon=True)
    with pytest.warns(UserWarning, match='zero residuals'):
        short.fit(y, fh=[1, 2, 3])
        wrong = short.predict_interval()
    pd.testing.assert_frame_equal(capped.predict_interval(), original.predict_interval())
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(wrong, original.predict_interval())


def test_deferred_horizon_caps_at_interval_time():
    y = pd.Series(np.arange(30, dtype=float) ** 1.1, name='sensor')
    original = ConformalIntervals(ContextForecaster(), initial_window=5).fit(y)
    capped = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                       cap_calibration_horizon=True).fit(y)
    pd.testing.assert_frame_equal(capped.predict_interval(fh=[1, 2, 4]),
                                  original.predict_interval(fh=[1, 2, 4]))
    assert capped.calibration_horizon_ == 5


def test_longer_interval_or_update_after_capped_fit_is_rejected():
    y = pd.Series(np.arange(30, dtype=float), name='sensor')
    capped = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                       cap_calibration_horizon=True).fit(y.iloc[:26], fh=[1, 2])
    with pytest.raises(ValueError, match='capped at 3'):
        capped.predict_interval(fh=[1, 2, 3])
    with pytest.raises(NotImplementedError, match='refit'):
        capped.update(y, update_params=True)


@pytest.mark.parametrize('options', [dict(sample_frac=0.5), dict(n_jobs=1)])
def test_native_fallbacks_stay_uncapped(options):
    y = pd.Series(np.arange(30, dtype=float), name='sensor')
    np.random.seed(7)
    original = ConformalIntervals(ContextForecaster(), initial_window=5, **options).fit(y, fh=[1, 2])
    np.random.seed(7)
    capped = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                       cap_calibration_horizon=True, **options).fit(y, fh=[1, 2])
    assert capped.calibration_horizon_ is None
    pd.testing.assert_frame_equal(capped.residuals_matrix_, original.residuals_matrix_)


def test_unsupported_callback_still_caps_serially():
    class Declines(RecordingBatch):
        @staticmethod
        def supports_forecaster(forecaster):
            return False

    callback = Declines()
    y = pd.Series(np.arange(30, dtype=float) ** 1.1, name='sensor')
    original, capped = _pair(y, [1, 2], callback=callback)
    assert callback.calls == []
    assert max(_requested_lengths(len(y))) == 3
    pd.testing.assert_frame_equal(capped.predict_interval(), original.predict_interval())


def test_cap_is_opt_in_and_cloned():
    y = pd.Series(np.arange(30, dtype=float), name='sensor')
    FitRecorder.fits.clear()
    default = BatchedConformalIntervals(FitRecorder(), initial_window=5,
                                        batch_predict=RecordingBatch()).fit(y, fh=[1])
    assert default.calibration_horizon_ is None
    assert max(_requested_lengths(len(y))) == 25
    assert BatchedConformalIntervals(ContextForecaster(), cap_calibration_horizon=True).clone(
    ).get_params()['cap_calibration_horizon'] is True
    with pytest.raises(TypeError, match='cap_calibration_horizon'):
        BatchedConformalIntervals(ContextForecaster(), cap_calibration_horizon=1)


@pytest.mark.parametrize('with_callback', [False, True])
def test_capped_fit_keeps_no_origin_forecasters(with_callback):
    """Each origin fit can hold its own model copy (sktime's Moirai 2 loads one per
    fit), so capped calibration must not retain them; only updates would use them."""
    y = pd.Series(np.arange(30, dtype=float) ** 1.1, name='sensor')
    callback = RecordingBatch() if with_callback else None
    _, capped = _pair(y, [1, 2], callback=callback)
    assert capped.forecasters_ == []
    uncapped = BatchedConformalIntervals(ContextForecaster(), initial_window=5,
                                         batch_predict=RecordingBatch()).fit(y, fh=[1, 2])
    assert len(uncapped.forecasters_) == 25
