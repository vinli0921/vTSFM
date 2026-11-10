"""Batch compatible calibration origins through sktime's loaded Chronos pipeline.

The pinned pipeline pads histories to complete input patches and forecasts whole
output patches before cropping. Matching native patch shapes and output-patch
schedules can share the same operation; horizons across patch boundaries remain
separate.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


@dataclass(frozen=True)
class ChronosCalibrationBatcher:
    """Use native Chronos batches where their full patch schedules match.

    Growing prefixes batch within the same native input-patch bucket. Complete
    output-patch counts must match, including long-horizon unrolling. Covariates,
    cross-learning and other forecasters keep their original serial forecasts.
    ``max_batch_points`` bounds padded inputs plus dense quantile outputs per
    optimized call; an oversized individual origin uses its native serial path.
    """

    max_batch_points: int = 1_048_576

    def __post_init__(self):
        if type(self.max_batch_points) is not int or self.max_batch_points < 1:
            raise ValueError('max_batch_points must be a positive integer')

    @staticmethod
    def supports_forecaster(forecaster):
        return (type(forecaster).__module__ == 'sktime.forecasting.chronos2'
                and type(forecaster).__name__ == 'Chronos2Forecaster'
                and forecaster.seed is not None
                and not forecaster._config['cross_learning'])

    def __call__(self, forecasters, future_indexes):
        if len(forecasters) != len(future_indexes):
            raise ValueError('Forecasters and calibration indexes must align')
        outputs = [None] * len(future_indexes)
        groups = {}
        for index, (forecaster, target) in enumerate(zip(forecasters, future_indexes)):
            key = self._key(forecaster, target)
            if key is None:
                outputs[index] = self._serial(forecaster, target)
            else:
                groups.setdefault(key, []).append(index)

        for indices in groups.values():
            batch = []
            for index in indices:
                if self._points(batch + [index], forecasters, future_indexes) > self.max_batch_points:
                    self._predict(batch, forecasters, future_indexes, outputs)
                    batch = []
                if self._points([index], forecasters, future_indexes) > self.max_batch_points:
                    outputs[index] = self._serial(forecasters[index], future_indexes[index])
                else:
                    batch.append(index)
            self._predict(batch, forecasters, future_indexes, outputs)
        return tuple(outputs)

    @staticmethod
    def _serial(forecaster, target):
        return forecaster.predict(fh=ForecastingHorizon(target, is_relative=False))

    @staticmethod
    def _key(forecaster, target):
        if not ChronosCalibrationBatcher.supports_forecaster(forecaster):
            return None
        if forecaster._cur_X is not None or forecaster._context.shape[0] != 1:
            return None
        relative = np.asarray(ForecastingHorizon(target, is_relative=False)
                              .to_relative(forecaster.cutoff))
        if not np.array_equal(relative, np.arange(1, len(target) + 1)):
            return None
        config = forecaster._config
        pipeline = forecaster.model_pipeline
        patch_size = getattr(pipeline, 'model_output_patch_size', 0)
        prediction_length = getattr(pipeline, 'model_prediction_length', 0)
        if config['cross_learning'] or patch_size < 1 or prediction_length < 1:
            return None
        width = ChronosCalibrationBatcher._context_width(forecaster)
        return (id(pipeline), forecaster._seed,
                config['context_length'] or pipeline.model_context_length,
                config['limit_prediction_length'], width,
                math.ceil(len(target) / patch_size))

    @staticmethod
    def _context_width(forecaster):
        width = forecaster._context.shape[-1]
        pipeline = forecaster.model_pipeline
        model_config = getattr(getattr(pipeline, 'model', None), 'chronos_config', None)
        input_patch = getattr(model_config, 'input_patch_size', 0)
        stride = getattr(model_config, 'input_patch_stride', 0)
        if input_patch > 0 and input_patch == stride:
            # The native dataset left-pads histories, then the model pads to a
            # complete patch. Within this bucket each origin therefore reaches
            # the same native patched width, without adding an extra token.
            width = math.ceil(width / input_patch) * input_patch
        return width

    @staticmethod
    def _points(indices, forecasters, targets):
        if not indices:
            return 0
        first = forecasters[indices[0]]
        patch = first.model_pipeline.model_output_patch_size
        horizon = math.ceil(max(len(targets[i]) for i in indices) / patch) * patch
        width = max(ChronosCalibrationBatcher._context_width(forecasters[i]) for i in indices)
        # Long horizons expand each origin into native quantile paths.
        paths = len(first.model_pipeline.quantiles) if horizon > first.model_pipeline.model_prediction_length else 1
        return len(indices) * paths * (width + len(first.model_pipeline.quantiles) * horizon)

    def _predict(self, indices, forecasters, targets, outputs):
        if not indices:
            return
        if len(indices) == 1:
            index = indices[0]
            outputs[index] = self._serial(forecasters[index], targets[index])
            return
        first = forecasters[indices[0]]
        from transformers import set_seed
        from vtsfm.recipes._chronos_ops import ChronosOriginMode

        set_seed(first._seed)
        pipeline = first.model_pipeline
        horizon = max(len(targets[index]) for index in indices)
        widths = [min(forecasters[index]._context.shape[-1], pipeline.model_context_length)
                  for index in indices]
        with ChronosOriginMode(pipeline, widths):
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
            outputs[index] = pd.Series(values[0, median, :len(targets[index])],
                                       index=targets[index], name=forecasters[index]._get_varnames()[0])
