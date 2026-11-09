"""Compare the opt-in entry point with the pinned, separately checked-out recipe."""

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_conformal_batching import ContextForecaster, RecordingBatch
from vtsfm.recipes.assetops import run_recipe

pd = pytest.importorskip('pandas')


@pytest.fixture
def composition(monkeypatch):
    root = os.environ.get('VTSFM_ASSETOPS_ROOT')
    if not root or not (Path(root) / 'src/servers/tsfm/engine/composition.py').is_file():
        pytest.skip('Set VTSFM_ASSETOPS_ROOT to the pinned AssetOpsBench checkout')
    monkeypatch.syspath_prepend(str(Path(root) / 'src'))
    # Only estimator resolution is replaced: the real upstream recipe's slicing,
    # calibration, imputation, anomaly labels and persistence execute unchanged.
    substrate = ModuleType('servers.tsfm.substrate')
    substrate.__path__ = []
    resolver = ModuleType('servers.tsfm.substrate.resolver')
    resolver.resolve = lambda spec: ContextForecaster()
    resolver.training_regime = lambda spec: 'zero_shot'
    substrate.resolver = resolver
    monkeypatch.setitem(sys.modules, 'servers.tsfm.substrate', substrate)
    monkeypatch.setitem(sys.modules, 'servers.tsfm.substrate.resolver', resolver)
    module = importlib.import_module('servers.tsfm.engine.composition')
    monkeypatch.setattr(module.uuid, 'uuid4', lambda: SimpleNamespace(hex='0123456789abcdef'))
    monkeypatch.setattr(module, '_now', lambda: 'fixed-time')
    return module


@pytest.mark.parametrize('impute', [None, 'zero', 'interpolate'])
@pytest.mark.parametrize('fh', [None, [1, 2, 3, 4]])
def test_real_recipe_keeps_labels_horizon_metadata_and_persistence(composition, impute, fh):
    y = pd.Series(np.sin(np.arange(45) / 4), name='sensor')
    if impute:
        y.iloc[10:12] = float('nan')
    y.iloc[-3] = 100
    recipe = {'task': 'tsfm_anomaly_detection', 'method': 'conformal',
              'estimator': {'sktime_class': 'test.ContextForecaster'},
              'conformal': {'coverage': 0.9}}
    if impute:
        recipe['impute'] = impute
    if fh:
        recipe['fh'] = fh
    options = dict(asset_id='sensor-1', parent_run_id='parent', scenario_id='scenario')
    records = []
    store = SimpleNamespace(put=lambda collection, value: records.append((collection, value)))
    expected = composition.run_recipe(store, y, recipe, **options)
    original_record = records.pop()
    callback = RecordingBatch()
    actual = run_recipe(store, y, recipe, batch_predict=callback, batch_size=7, **options)
    assert actual == expected
    assert records.pop() == original_record
    assert all(len(call) <= 7 for call in callback.calls)
    assert max(len(index) for call in callback.calls for _, index in call) > len(fh or range(9))


def test_other_recipe_tasks_delegate_without_reinterpreting_them(composition, monkeypatch):
    calls = []
    monkeypatch.setattr(composition, 'run_recipe', lambda *args, **kwargs: calls.append((args, kwargs)) or 'original')
    recipe = {'task': 'tsfm_forecasting'}
    assert run_recipe(None, [1, 2], recipe, batch_predict=None) == 'original'
    assert calls[0][0] == (None, [1, 2], recipe)


def test_changed_upstream_recipe_requires_revalidation(composition, monkeypatch):
    def changed_recipe(*args, **kwargs):
        return 'changed'
    monkeypatch.setattr(composition, '_conformal_ad', changed_recipe)
    recipe = {'task': 'tsfm_anomaly_detection', 'method': 'conformal'}
    with pytest.raises(NotImplementedError, match='revalidate'):
        run_recipe(None, [1, 2], recipe, batch_predict=RecordingBatch())
