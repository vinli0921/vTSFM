"""Opt-in AssetOpsBench conformal calibration without modifying its installation.

The conformal branch follows AssetOpsBench e844f6468c6c68b328022e7ca8e31da140842cc7
(Apache-2.0). Only the interval estimator is replaced;
imputation, horizon policy, anomaly decisions and result records are preserved.
"""

import hashlib
import inspect


_CONFORMAL_SOURCE_SHA256 = '17ddb50cd3bc6e3c3ff19b4d5ddf03d4852fa024061c0096e722c92441a392fe'


def run_recipe(store, y, recipe, *, batch_predict, batch_size=64, asset_id='asset',
               parent_run_id=None, scenario_id=None):
    """AssetOps run_recipe entry point with an explicit calibration batch callback.

    Install AssetOpsBench separately. Other recipe tasks use its original runner.
    The batched path currently supports zero-shot univariate conformal anomaly
    recipes; it retains sktime's default ``empirical`` interval method.
    """
    from servers.tsfm.engine import composition

    options = dict(asset_id=asset_id, parent_run_id=parent_run_id, scenario_id=scenario_id)
    if recipe.get('task') != 'tsfm_anomaly_detection' or recipe.get('method') != 'conformal':
        return composition.run_recipe(store, y, recipe, **options)
    source = inspect.getsource(composition._conformal_ad).encode()
    if hashlib.sha256(source).hexdigest() != _CONFORMAL_SOURCE_SHA256:
        raise NotImplementedError('AssetOps conformal recipe changed; revalidate the batching integration')

    import numpy as np
    import pandas as pd
    from servers.tsfm.substrate import resolver
    from .conformal import BatchedConformalIntervals

    y = pd.Series(np.asarray(y, float)) if not isinstance(y, pd.Series) else y
    impute = recipe.get('impute') or (recipe.get('anomaly') or {}).get('impute')
    if impute:
        y = composition._impute(y, impute)
    spec = recipe.get('estimator')
    if not spec:
        raise ValueError("conformal AD needs an 'estimator' (a forecaster card)")
    card = (composition.model_store_get(store, spec['model_id']) or spec
            if spec.get('model_id') and store is not None else spec)
    merged = {**card, 'params': {**(card.get('params') or {}), **(spec.get('params') or {})}}
    regime = resolver.training_regime(merged)
    if regime != 'zero_shot':
        raise ValueError('Batched calibration requires a zero-shot forecaster')
    coverage = float((recipe.get('conformal') or {}).get('coverage', 0.9))
    fh = recipe.get('fh') or list(range(1, max(2, len(y) // 5) + 1))
    horizon = len(fh)
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    intervals = BatchedConformalIntervals(resolver.resolve(merged), batch_predict, batch_size)
    intervals.fit(y_train, fh=fh)
    prediction = intervals.predict_interval(coverage=coverage)
    low = np.asarray(prediction.iloc[:, 0])[:horizon]
    high = np.asarray(prediction.iloc[:, 1])[:horizon]
    actual = np.asarray(y_test)[:horizon]
    outside = (actual < low) | (actual > high)
    labels = np.zeros(len(y), dtype=int)
    labels[len(y) - horizon:][outside] = 1
    indices = np.where(labels == 1)[0].tolist()
    count = int(labels.sum())

    run_id = f'run:{composition.uuid.uuid4().hex[:10]}'
    record = {
        '_id': run_id, 'run_id': run_id, 'parent_run_id': parent_run_id,
        'asset_id': asset_id, 'scenario_id': scenario_id,
        'task': 'tsfm_anomaly_detection', 'method': 'conformal', 'recipe': recipe,
        'training_regime': regime, 'coverage': coverage, 'n_anomalies': count,
        'n_observations': len(y), 'anomaly_indices': indices[:200],
        'created_at': composition._now(),
    }
    if store is not None:
        store.put(composition.RUNS, record)
    return {
        'run_id': run_id, 'task': 'tsfm_anomaly_detection', 'method': 'conformal',
        'n_anomalies': count, 'n_observations': len(y),
        'anomaly_indices_head': indices[:20], 'labels': labels.tolist(),
        'training_regime': regime, 'coverage': coverage, 'parent_run_id': parent_run_id,
    }
