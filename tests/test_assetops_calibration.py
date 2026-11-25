"""Run the original recipe and an isolated copy with the optional factory hook."""

from functools import partial
import importlib
import importlib.util
from importlib import resources
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_conformal_batching import (
    BatchedConformalIntervals, ContextForecaster, RecordingBatch,
)

pd = pytest.importorskip('pandas')


@pytest.fixture
def compositions(monkeypatch, tmp_path):
    root = os.environ.get('VTSFM_ASSETOPS_ROOT')
    relative = Path('src/servers/tsfm/engine/composition.py')
    if not root or not (Path(root) / relative).is_file():
        pytest.skip('Set VTSFM_ASSETOPS_ROOT to the pinned AssetOpsBench checkout')
    monkeypatch.syspath_prepend(str(Path(root) / 'src'))
    # Estimator resolution is deterministic; recipe orchestration is actual
    # upstream code. Neither installed package is modified by this fixture.
    substrate = ModuleType('servers.tsfm.substrate')
    substrate.__path__ = []
    resolver = ModuleType('servers.tsfm.substrate.resolver')
    resolver.resolve = lambda spec: ContextForecaster()
    resolver.training_regime = lambda spec: 'zero_shot'
    substrate.resolver = resolver
    monkeypatch.setitem(sys.modules, 'servers.tsfm.substrate', substrate)
    monkeypatch.setitem(sys.modules, 'servers.tsfm.substrate.resolver', resolver)
    original = importlib.import_module('servers.tsfm.engine.composition')
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(Path(root) / relative, target)
    patch = resources.files('vtsfm.recipes').joinpath('assetops-conformal-factory.patch').read_text()
    result = subprocess.run(['patch', '--batch', '-p1'], input=patch,
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    spec = importlib.util.spec_from_file_location('servers.tsfm.engine.patched_composition', target)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    for module in [original, patched]:
        monkeypatch.setattr(module.uuid, 'uuid4', lambda: SimpleNamespace(hex='0123456789abcdef'))
        monkeypatch.setattr(module, '_now', lambda: 'fixed-time')
    return original, patched


@pytest.mark.parametrize('impute', [None, 'zero', 'interpolate'])
@pytest.mark.parametrize('fh', [None, [1, 2, 3, 4]])
@pytest.mark.parametrize('mode', ['default', 'serial', 'batched', 'capped', 'capped-batched'])
def test_real_recipe_preserves_labels_horizons_metadata_and_persistence(compositions, impute, fh, mode):
    original, patched = compositions
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
    expected = original.run_recipe(store, y, recipe, **options)
    original_record = records.pop()
    callback = RecordingBatch()
    capped = mode.startswith('capped')
    if mode != 'default':
        options['conformal_factory'] = partial(BatchedConformalIntervals, batch_size=7,
                                               batch_predict=callback if mode.endswith('batched') else None,
                                               cap_calibration_horizon=capped)
    actual = patched.run_recipe(store, y, recipe, **options)
    assert actual == expected
    assert records.pop() == original_record
    longest = max((len(index) for call in callback.calls for _, index in call), default=0)
    if mode.endswith('batched'):
        assert callback.calls and all(len(call) <= 7 for call in callback.calls)
        # The recipe's default horizon is len(y) // 5 = 9 steps.
        horizon = len(fh or range(9))
        assert longest == horizon + 1 if capped else longest > horizon + 1
    else:
        assert callback.calls == []


def test_optional_factory_does_not_change_the_existing_detector_path(compositions, monkeypatch):
    original, patched = compositions

    class Detector:
        def fit(self, y):
            self.length = len(y)
            return self

        def predict(self, y):
            return np.zeros(self.length, dtype=int)

    def unexpected_factory(forecaster):
        pytest.fail('A conformal factory must not run for another recipe method')

    resolver = sys.modules['servers.tsfm.substrate.resolver']
    monkeypatch.setattr(resolver, 'resolve', lambda spec: Detector())
    # The minimal source checkout lacks the separate recipe-audit service.
    for module in [original, patched]:
        monkeypatch.setattr(module, '_validate_blocks', lambda recipe: {})
    recipe = {'task': 'tsfm_anomaly_detection', 'method': 'detector',
              'estimator': {'sktime_class': 'test.Detector'}}
    assert patched.run_recipe(None, [1, 2, 3], recipe, conformal_factory=unexpected_factory) == (
        original.run_recipe(None, [1, 2, 3], recipe))


def test_checkout_helper_connects_actual_recipe_and_resolver(tmp_path):
    root = os.environ.get('VTSFM_ASSETOPS_ROOT')
    relatives = [Path('src/servers/tsfm/engine/composition.py'),
                 Path('src/servers/tsfm/substrate/resolver.py')]
    if not root or any(not (Path(root) / path).is_file() for path in relatives):
        pytest.skip('Set VTSFM_ASSETOPS_ROOT to the pinned AssetOpsBench checkout')
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(root) / relative, target)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'src'], check=True)
    # A fresh interpreter uses the real resolver rather than the orchestration
    # fixture's deterministic estimator substitute.
    code = '''
from functools import partial
import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from vtsfm.recipes.assetops import patch_assetops
from vtsfm.recipes.conformal import BatchedConformalIntervals
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
from servers.tsfm.engine import composition
from servers.tsfm.substrate import resolver
assert Path(resolver.__file__) == root / 'src/servers/tsfm/substrate/resolver.py'
y = pd.Series(np.sin(np.arange(45) / 4))
y.iloc[-3] = 100
recipe = {'task': 'tsfm_anomaly_detection', 'method': 'conformal',
          'estimator': {'sktime_class': 'sktime.forecasting.naive.NaiveForecaster'},
          'conformal': {'coverage': 0.9}, 'fh': [1, 2, 3, 4]}
expected = composition.run_recipe(None, y, recipe)
assert 'Applied' in patch_assetops(root, apply=True)
composition = importlib.reload(composition)
calls = []
def forecast(forecasters, targets):
    calls.append(len(forecasters))
    return tuple(fc.predict(fh=ForecastingHorizon(index, is_relative=False))
                 for fc, index in zip(forecasters, targets))
factory = partial(BatchedConformalIntervals, batch_predict=forecast, batch_size=7)
actual = composition.run_recipe(None, y, recipe, conformal_factory=factory)
actual.pop('run_id')
expected.pop('run_id')
assert actual == expected
assert calls and max(calls) == 7 and calls[-1] < 7
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
