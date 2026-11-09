"""Optional batch bridge for sktime 1.2's fitted Chronos-2 forecasters.

Unqualified contexts/horizons stay in separate calls. Enabling either padding
capability requires model-level parity evidence; the integration never infers
prefix invariance from a forecasting API's signature.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


@dataclass(frozen=True)
class ChronosCalibrationBatcher:
    """Run independent contexts through one already-loaded Chronos pipeline.

    The defaults preserve exact context widths and requested horizons. A single
    series' growing-origin calibration consequently remains unbatched until
    ragged-context and horizon-prefix equivalence have both been qualified.
    ``max_batch_points`` bounds input plus dense output points per model call.
    """

    context_padding_invariant: bool = False
    horizon_prefix_invariant: bool = False
    max_batch_points: int = 1_048_576

    def __post_init__(self):
        if (type(self.context_padding_invariant) is not bool
                or type(self.horizon_prefix_invariant) is not bool):
            raise TypeError('Batching capabilities must be bools')
        if type(self.max_batch_points) is not int or self.max_batch_points < 1:
            raise ValueError('max_batch_points must be a positive integer')

    def __call__(self, forecasters, targets):
        if len(forecasters) != len(targets):
            raise ValueError('Forecasters and calibration indexes must align')
        outputs = [None] * len(targets)
        groups = {}
        for index, (forecaster, target) in enumerate(zip(forecasters, targets)):
            if (type(forecaster).__module__ != 'sktime.forecasting.chronos2'
                    or type(forecaster).__name__ != 'Chronos2Forecaster'):
                raise ValueError('This batch bridge requires sktime Chronos2Forecaster')
            if forecaster._cur_X is not None or forecaster._context.shape[0] != 1:
                raise ValueError('Chronos calibration requires univariate contexts without covariates')
            relative = np.asarray(ForecastingHorizon(target, is_relative=False)
                                  .to_relative(forecaster.cutoff))
            if not np.array_equal(relative, np.arange(1, len(target) + 1)):
                raise ValueError('Chronos calibration requires consecutive future horizons')
            config = forecaster._config
            if config['cross_learning']:
                raise ValueError('Calibration origins must remain independent; cross_learning must be false')
            pipeline = forecaster.model_pipeline
            context_length = config['context_length'] or pipeline.model_context_length
            key = (id(pipeline), forecaster._seed, context_length,
                   config['limit_prediction_length'],
                   None if self.context_padding_invariant else forecaster._context.shape[-1],
                   None if self.horizon_prefix_invariant else len(target))
            groups.setdefault(key, []).append(index)

        for indices in groups.values():
            batch = []
            for index in indices:
                candidate = batch + [index]
                points = len(candidate) * (
                    max(forecasters[i]._context.size for i in candidate)
                    + len(forecasters[index].model_pipeline.quantiles)
                    * max(len(targets[i]) for i in candidate))
                if points > self.max_batch_points:
                    if not batch:
                        raise ValueError('One calibration origin exceeds max_batch_points')
                    self._predict(batch, forecasters, targets, outputs)
                    batch = [index]
                    if (forecasters[index]._context.size
                            + len(forecasters[index].model_pipeline.quantiles) * len(targets[index])
                            > self.max_batch_points):
                        raise ValueError('One calibration origin exceeds max_batch_points')
                else:
                    batch = candidate
            if batch:
                self._predict(batch, forecasters, targets, outputs)
        return tuple(outputs)

    def _predict(self, indices, forecasters, targets, outputs):
        if len(indices) == 1:
            index = indices[0]
            outputs[index] = forecasters[index].predict(
                fh=ForecastingHorizon(targets[index], is_relative=False))
            return
        first = forecasters[indices[0]]
        from transformers import set_seed

        set_seed(first._seed)
        pipeline = first.model_pipeline
        horizon = max(len(targets[index]) for index in indices)
        predictions = pipeline.predict(
            [{'target': forecasters[index]._context} for index in indices],
            prediction_length=horizon, batch_size=len(indices),
            context_length=first._config['context_length'] or pipeline.model_context_length,
            cross_learning=False,
            limit_prediction_length=first._config['limit_prediction_length'])
        if len(predictions) != len(indices) or 0.5 not in pipeline.quantiles:
            raise ValueError('Chronos batch outputs must align and include the median')
        median = pipeline.quantiles.index(0.5)
        for index, prediction in zip(indices, predictions):
            values = np.asarray(prediction)
            if (values.ndim != 3 or values.shape[:2] != (1, len(pipeline.quantiles))
                    or values.shape[-1] < horizon):
                raise ValueError('Chronos batch output must be [1, quantiles, horizon]')
            target = targets[index]
            outputs[index] = pd.Series(values[0, median, :len(target)],
                                       index=target, name=forecasters[index]._get_varnames()[0])
