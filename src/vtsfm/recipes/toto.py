"""Batch sktime Toto-2 origins with matching native padding and decode schedules."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


@dataclass(frozen=True)
class TotoCalibrationBatcher:
    """Reuse the cached native model and sktime's exact context alignment.

    Group only equal aligned contexts, mask paths, seeds and full output-patch
    counts. Other forecasters or unsupported shapes retain their serial calls.
    The point budget bounds aligned input and dense quantile output values;
    individual oversized origins use the native serial path.
    """

    max_batch_points: int = 1_048_576

    def __post_init__(self):
        if type(self.max_batch_points) is not int or self.max_batch_points < 1:
            raise ValueError('max_batch_points must be a positive integer')

    @staticmethod
    def supports_forecaster(forecaster):
        return (type(forecaster).__module__ == 'sktime.forecasting.toto2'
                and type(forecaster).__name__ == 'Toto2Forecaster'
                and forecaster.seed is not None)

    @staticmethod
    def _serial(forecaster, target):
        return forecaster.predict(fh=ForecastingHorizon(target, is_relative=False))

    def __call__(self, forecasters, future_indexes):
        if len(forecasters) != len(future_indexes):
            raise ValueError('Forecasters and calibration indexes must align')
        outputs = [None] * len(future_indexes)
        groups = {}
        for index, (fc, target) in enumerate(zip(forecasters, future_indexes)):
            relative = np.asarray(ForecastingHorizon(target, is_relative=False).to_relative(fc.cutoff))
            if (not self.supports_forecaster(fc)
                    or fc._cur_X is not None or fc._inputs_['target'].shape[:2] != (1, 1)
                    or not np.array_equal(relative, np.arange(1, len(target) + 1))):
                outputs[index] = self._serial(fc, target)
                continue
            # The native sktime path uses this same cache and alignment helper.
            from sktime.forecasting.toto2 import _CachedToto2Forecaster
            import torch

            torch.manual_seed(fc._seed)
            model = _CachedToto2Forecaster(key=str((fc.model_path, fc._device)),
                model_path=fc.model_path, device=fc._device).load_from_checkpoint()
            patch = model.config.patch_size
            values, mask, missing = fc._align_context(
                fc._inputs_['target'], fc._inputs_['target_mask'], patch)
            horizon = math.ceil(len(target) / patch) * patch
            points = values.numel() + len(model.output_head.knots) * horizon
            if points > self.max_batch_points:
                outputs[index] = self._serial(fc, target)
                continue
            key = (id(model), fc._seed, fc.decode_block_size, values.shape[-1], missing, horizon)
            groups.setdefault(key, []).append((index, fc, values, mask, model, points))

        for entries in groups.values():
            batch = []
            for entry in entries:
                if sum(row[-1] for row in batch) + entry[-1] > self.max_batch_points:
                    self._predict(batch, future_indexes, outputs)
                    batch = []
                batch.append(entry)
            self._predict(batch, future_indexes, outputs)
        return tuple(outputs)

    def _predict(self, entries, targets, outputs):
        if len(entries) == 1:
            index, fc, *_ = entries[0]
            outputs[index] = self._serial(fc, targets[index])
            return
        import torch
        from ._toto_linear import _OriginLinearMode

        first = entries[0][1]
        model = entries[0][4]
        torch.manual_seed(first._seed)
        horizon = max(len(targets[row[0]]) for row in entries)
        values = torch.cat([row[2] for row in entries], dim=0)
        masks = torch.cat([row[3] for row in entries], dim=0)
        ids = torch.cat([row[1]._inputs_['series_ids'] for row in entries], dim=0)
        with torch.no_grad(), _OriginLinearMode(len(entries)):
            predictions = model.forecast({'target': values, 'target_mask': masks, 'series_ids': ids},
                horizon=horizon, decode_block_size=first.decode_block_size,
                has_missing_values=entries[0][2].shape[-1] != first._inputs_['target'].shape[-1])
        array = predictions.detach().cpu().numpy()
        knots = model.output_head.knots
        if array.ndim != 4 or array.shape[:3] != (len(knots), len(entries), 1) or array.shape[-1] < horizon:
            raise ValueError('Toto batch output must be [quantiles, origins, 1, horizon]')
        median = knots.index(0.5)
        for row, (index, fc, *_) in enumerate(entries):
            outputs[index] = pd.Series(array[median, row, 0, :len(targets[index])],
                                       index=targets[index], name=fc._cur_y.columns[0])
