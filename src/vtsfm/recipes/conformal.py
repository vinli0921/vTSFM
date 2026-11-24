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

    With no callback, or with covariates, origin sampling, explicit ``n_jobs``
    or updates, sktime's original implementation runs. Existing constructor
    arguments retain their original order and defaults.
    """

    def __init__(self, forecaster, method='empirical', initial_window=None,
                 sample_frac=None, verbose=False, n_jobs=None, *,
                 batch_predict=None, batch_size=64):
        if batch_predict is not None and not callable(batch_predict):
            raise TypeError('batch_predict must be callable')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or not 1 <= batch_size <= 256:
            raise ValueError('batch_size must be an integer in [1,256]')
        self.batch_predict = batch_predict
        self.batch_size = batch_size
        super().__init__(forecaster=forecaster, method=method,
                         initial_window=initial_window, sample_frac=sample_frac,
                         verbose=verbose, n_jobs=n_jobs)

    def _compute_sliding_residuals(self, y, X, forecaster, initial_window,
                                   sample_frac, update=False):
        supports = getattr(self.batch_predict, 'supports_forecaster', None)
        if (self.batch_predict is None or X is not None or update
                or sample_frac is not None or self.n_jobs is not None or isinstance(y.index, pd.MultiIndex)
                or (supports is not None and not supports(forecaster))):
            return super()._compute_sliding_residuals(
                y, X, forecaster, initial_window, sample_frac, update)
        y = convert_to(y, 'pd.Series')
        first = self._parse_initial_window(y, initial_window)
        index = y.iloc[first:].index
        residuals = pd.DataFrame(index=index, columns=index, dtype=float)

        for start in range(first, len(y), self.batch_size):
            origins = range(start, min(start + self.batch_size, len(y)))
            clones, targets = [], []
            for origin in origins:
                target = y.iloc[origin:]
                fitted = forecaster.clone().fit(y.iloc[:origin], fh=target.index)
                clones.append(fitted)
                targets.append(target)
            result = self.batch_predict(tuple(clones), tuple(target.index for target in targets))
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
                # Native sktime updates reuse these fitted origin forecasters.
                self.forecasters_.append({'id': str(y.index[origin]), 'forecaster': fitted})
        return residuals
