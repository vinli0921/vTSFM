# Adapted from sktime 1.2.0's conformal calibration loop.
# Source: sktime/forecasting/conformal.py (authors fkiraly and bethrice44).
# BSD 3-Clause License
#
# Copyright (c) 2019 - present, The sktime developers.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


"""Bounded calibration batches for sktime's existing interval calculations.

Only the sliding-origin forecast loop changes. sktime retains interval quantiles,
coverage, the final fitted forecast, and its signed-residual convention.

Lead indexing (sktime 1.2.0): residual row ``origin`` comes from a fit on
``y.iloc[:origin]``, so column ``origin`` is lead 1. ``_predict_interval_series``
reads ``np.diagonal(residuals, offset=k)`` for relative step ``k``, which is
lead ``k + 1``. Intervals for ``fh = 1..H`` therefore read leads ``2..H + 1``;
lead 1 and leads beyond ``H + 1`` are computed but never read.
"""

from numbers import Integral

import numpy as np
import pandas as pd
from sktime.datatypes import convert_to
from sktime.forecasting.conformal import ConformalIntervals


class BatchedConformalIntervals(ConformalIntervals):
    """Use an explicit point-forecast batch callback during initial calibration.

    ``batch_predict(forecasters, future_indexes)`` receives fitted, independent clones
    and their original future indexes. It must return one point-forecast Series
    per clone, with exactly the target index. Contexts and horizons can differ;
    the callback owns any model-specific collation or grouping restrictions.

    With ``cap_calibration_horizon=True`` (#19), each calibration origin
    forecasts only ``max(fh) + 1`` steps, the longest lead any interval method
    reads, instead of the whole remaining history. Residual-matrix entries
    beyond that lead stay NaN. The cap works with or without a callback; it
    requires a known positive horizon, and a later interval request that needs a
    longer lead raises instead of reading the missing residuals.

    With no callback and no cap, or with covariates, origin sampling, explicit
    ``n_jobs``, updates or a multi-index series, sktime's original uncapped
    implementation runs. ``update`` after a capped fit is rejected, so a capped
    fit keeps no per-origin forecasters in ``forecasters_``. Existing
    constructor arguments retain their original order and defaults.
    """

    def __init__(self, forecaster, method='empirical', initial_window=None,
                 sample_frac=None, verbose=False, n_jobs=None, *,
                 batch_predict=None, batch_size=64, cap_calibration_horizon=False):
        if batch_predict is not None and not callable(batch_predict):
            raise TypeError('batch_predict must be callable')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or not 1 <= batch_size <= 256:
            raise ValueError('batch_size must be an integer in [1,256]')
        if not isinstance(cap_calibration_horizon, bool):
            raise TypeError('cap_calibration_horizon must be a bool')
        self.batch_predict = batch_predict
        self.batch_size = batch_size
        self.cap_calibration_horizon = cap_calibration_horizon
        super().__init__(forecaster=forecaster, method=method,
                         initial_window=initial_window, sample_frac=sample_frac,
                         verbose=verbose, n_jobs=n_jobs)

    def _needed_lead(self, fh):
        """Longest residual lead the intervals for ``fh`` read (see module docstring)."""
        steps = np.asarray(fh.to_relative(self.cutoff))
        return int(steps.max()) + 1 if len(steps) and steps.max() >= 1 else None

    def _compute_sliding_residuals(self, y, X, forecaster, initial_window,
                                   sample_frac, update=False):
        # Both the fit-time and predict-time residual paths set self._fh first.
        horizon = (self._needed_lead(self._fh)
                   if self.cap_calibration_horizon and self._fh is not None else None)
        batch_predict = self.batch_predict
        supports = getattr(batch_predict, 'supports_forecaster', None)
        if supports is not None and not supports(forecaster):
            batch_predict = None
        self.calibration_horizon_ = None
        if ((batch_predict is None and horizon is None) or X is not None or update
                or sample_frac is not None or self.n_jobs is not None or isinstance(y.index, pd.MultiIndex)):
            return super()._compute_sliding_residuals(
                y, X, forecaster, initial_window, sample_frac, update)
        self.calibration_horizon_ = horizon
        y = convert_to(y, 'pd.Series')
        first = self._parse_initial_window(y, initial_window)
        index = y.iloc[first:].index
        residuals = pd.DataFrame(index=index, columns=index, dtype=float)

        for start in range(first, len(y), self.batch_size):
            origins = range(start, min(start + self.batch_size, len(y)))
            clones, targets = [], []
            for origin in origins:
                target = y.iloc[origin:] if horizon is None else y.iloc[origin:origin + horizon]
                fitted = forecaster.clone().fit(y.iloc[:origin], fh=target.index)
                clones.append(fitted)
                targets.append(target)
            if batch_predict is None:
                # Serial capped calibration: sktime's own fit/predict_residuals per origin.
                predictions = (None,) * len(clones)
            else:
                result = batch_predict(tuple(clones), tuple(target.index for target in targets))
                try:
                    iterator = iter(result)
                except TypeError:
                    predictions = ()
                else:
                    predictions = tuple(iterator)
            if len(predictions) != len(clones):
                predictions = (None,) * len(clones)
            for origin, fitted, target, prediction in zip(origins, clones, targets, predictions):
                values = None
                if isinstance(prediction, pd.Series) and prediction.index.equals(target.index):
                    try:
                        values = prediction.to_numpy(dtype=float)
                    except (TypeError, ValueError):
                        pass
                if values is None or not np.isfinite(values).all():
                    residuals.loc[y.index[origin]] = fitted.predict_residuals(target)
                else:
                    if (pd.api.types.is_integer_dtype(target.dtype)
                            and pd.api.types.is_integer_dtype(prediction.dtype)):
                        # Subtract Python integers before rounding or fixed-width overflow.
                        difference = target.astype(object) - prediction.astype(object)
                        difference = difference.to_numpy(dtype=float, na_value=np.nan)
                    else:
                        difference = target.to_numpy(dtype=float) - values
                    residuals.loc[y.index[origin], target.index] = difference
                # Native sktime updates reuse these fitted origin forecasters. Capped fits
                # reject updates, so they keep none: wrappers that load a model on every
                # fit (sktime's Moirai 2) would otherwise hold one GPU copy per origin.
                if horizon is None:
                    self.forecasters_.append({'id': str(y.index[origin]), 'forecaster': fitted})
        return residuals

    def _update(self, y, X=None, update_params=True):
        # Native updates extend rows with origin forecasters fitted for the capped
        # horizon, which would mix capped and uncapped residuals. Refit instead.
        if update_params and getattr(self, 'calibration_horizon_', None) is not None:
            raise NotImplementedError('update after capped calibration is unsupported; refit instead')
        return super()._update(y, X, update_params=update_params)

    def _predict_interval_series(self, fh, coverage, y_pred):
        horizon = getattr(self, 'calibration_horizon_', None)
        if self.fh_early_ and horizon is not None:
            needed = self._needed_lead(fh)
            if needed is not None and needed > horizon:
                raise ValueError(f'Interval needs residual lead {needed}, but calibration was '
                                 f'capped at {horizon}; refit with this horizon')
        return super()._predict_interval_series(fh, coverage, y_pred)
