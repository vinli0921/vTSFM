"""Opt-in real-checkpoint checks for capped conformal calibration (#19).

Models come from AssetOps catalog cards in the profiling harness's
``recipe_estimator`` style, not vTSFM runtimes. Run one adapter per Python
environment:

    VTSFM_RECIPE_MODEL_TEST=moirai2|timesfm2|toto2   (required)
    VTSFM_RECIPE_DEVICE=cuda                         (default; ``cpu`` also works)
    VTSFM_ASSETOPS_ROOT=/path/to/AssetOpsBench       (recipe label check only)
    VTSFM_RECIPE_COMPARE_UNCAPPED=1                  (also run the uncapped Moirai recipe)
    VTSFM_RECIPE_CUTOFFS=204,1000,2048               (optional subset of cutoffs)
    VTSFM_RECIPE_RESULTS=results/issue19             (JSON output directory, git-ignored)

Each run writes ``<results>/<model>.json`` (``<model>-uncapped.json`` with
``VTSFM_RECIPE_COMPARE_UNCAPPED=1``) with the card, cutoffs, differences, labels,
package versions, device, git state and wall times, including failures.
"""

from functools import partial
import hashlib
import importlib
import importlib.metadata
import importlib.util
from importlib import resources
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

ADAPTER = os.environ.get('VTSFM_RECIPE_MODEL_TEST')
if ADAPTER not in ('moirai2', 'timesfm2', 'toto2'):
    pytest.skip('Set VTSFM_RECIPE_MODEL_TEST to moirai2, timesfm2 or toto2', allow_module_level=True)

torch = pytest.importorskip('torch')
pd = pytest.importorskip('pandas')
pytest.importorskip('sktime')
from chiller6_fixture import load_series  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get('VTSFM_RECIPE_DEVICE', 'cuda')
RESULTS = Path(os.environ.get('VTSFM_RECIPE_RESULTS', REPOSITORY / 'results' / 'issue19'))
HORIZON = 96
CAPPED = HORIZON + 1  # sktime's intervals for fh=1..96 read residual leads up to 97.
# TimesFM's pretrained point head emits config.horizon_length=128 points per call,
# and sktime rejects longer requests, so 128 is that card's longest forecast.
LONG = 128 if ADAPTER == 'timesfm2' else 1024
TOLERANCE = dict(atol=1e-4, rtol=1e-4)
# The recipe's first calibration origin, a middle one, and the full 2,048-point
# training history used for the final forecast.
CUTOFFS = [int(value) for value in os.environ.get('VTSFM_RECIPE_CUTOFFS', '204,1000,2048').split(',')]
RECORD = {'model': ADAPTER, 'prefix': [], 'recipe': {}}


def card(adapter=ADAPTER):
    """The harness's catalog card, with only the device configurable."""
    params = {
        'timesfm2': {'model_path': 'google/timesfm-2.5-200m-transformers',
                     'device_map': DEVICE, 'dtype': torch.float32},
        'toto2': {'model_path': 'Datadog/Toto-2.0-313m', 'device': DEVICE, 'seed': 0},
        # The estimator's default context_length is 200; the harness used 2048.
        # sktime's vendored Moirai2Forecast needs the separate recipe environment.
        'moirai2': {'checkpoint_path': 'Salesforce/moirai-2.0-R-small', 'context_length': 2048,
                    'map_location': DEVICE},
    }[adapter]
    path = {
        'timesfm2': 'sktime.forecasting.timesfm2.TimesFM2Forecaster',
        'toto2': 'sktime.forecasting.toto2.Toto2Forecaster',
        'moirai2': 'sktime.forecasting.moirai2.Moirai2Forecaster',
    }[adapter]
    spec = {'name': adapter, 'sktime_class': path, 'params': params}
    if adapter == 'toto2':
        spec['training_regime'] = 'zero_shot'
    return spec


def build(spec):
    """Equivalent to AssetOps ``resolver.resolve`` for these flat cards."""
    module, name = spec['sktime_class'].rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(**spec['params'])


def forecast(context, steps):
    fitted = build(card()).fit(context, fh=list(range(1, steps + 1)))
    return fitted.predict().to_numpy(dtype=float)


def _git(*arguments):
    result = subprocess.run(['git', '-C', str(REPOSITORY), *arguments], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def _environment():
    versions = {}
    for package in ['torch', 'sktime', 'numpy', 'pandas', 'transformers', 'uni2ts', 'gluonts',
                    'toto-2', 'toto-models', 'skpro', 'huggingface-hub']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    diff = _git('diff', 'HEAD')
    return {
        'versions': versions, 'python': platform.python_version(), 'host': platform.node(),
        'device': DEVICE, 'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git': {'commit': (_git('rev-parse', 'HEAD') or '').strip() or None,
                'dirty': bool((_git('status', '--porcelain') or '').strip()),
                'diff_sha256': hashlib.sha256(diff.encode()).hexdigest() if diff else None},
    }


@pytest.fixture(scope='module', autouse=True)
def results_file():
    spec = card()
    RECORD.update(card={**spec, 'params': {key: str(value) if key == 'dtype' else value
                                           for key, value in spec['params'].items()}},
                  tolerance=TOLERANCE, horizon=HORIZON,
                  started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **_environment())
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    # The optional uncapped Moirai comparison gets its own file.
    suffix = '-uncapped' if os.environ.get('VTSFM_RECIPE_COMPARE_UNCAPPED') == '1' else ''
    path = RESULTS / f'{ADAPTER}{suffix}.json'
    path.write_text(json.dumps(RECORD, indent=2, default=str) + '\n')
    print(f'wrote {path}')


@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_capped_calibration_forecast_matches_long_forecast_prefix(cutoff):
    entry = {'cutoff': cutoff, 'short_steps': CAPPED, 'long_steps': LONG, 'compared_steps': CAPPED}
    RECORD['prefix'].append(entry)
    start = time.perf_counter()
    try:
        context = load_series().iloc[:cutoff]
        short = forecast(context, CAPPED)
        if ADAPTER == 'timesfm2':
            with pytest.raises(ValueError, match='exceeds TimesFM model capacity'):
                forecast(context, 1024)
            entry['rejects_1024'] = True
        long = forecast(context, LONG)
        assert short.shape == (CAPPED,) and np.isfinite(short).all()
        absolute = np.abs(short - long[:CAPPED])
        entry.update(max_abs_diff=float(absolute.max()),
                     max_rel_diff=float((absolute / np.maximum(np.abs(long[:CAPPED]), 1e-12)).max()))
        print(f'{ADAPTER} cutoff={cutoff} steps={CAPPED} vs {LONG} max_abs_diff={entry["max_abs_diff"]:.3e}')
        np.testing.assert_allclose(short, long[:CAPPED], **TOLERANCE)
        entry['passed'] = True
    except BaseException as error:
        entry.update(passed=False, error=f'{type(error).__name__}: {error}'[:2000])
        raise
    finally:
        entry['wall_s'] = round(time.perf_counter() - start, 3)


@pytest.fixture
def composition(tmp_path):
    root = os.environ.get('VTSFM_ASSETOPS_ROOT')
    relative = Path('src/servers/tsfm/engine/composition.py')
    if not root or not (Path(root) / relative).is_file():
        RECORD['recipe']['skipped'] = 'VTSFM_ASSETOPS_ROOT not set'
        pytest.skip('Set VTSFM_ASSETOPS_ROOT to the pinned AssetOpsBench checkout')
    RECORD['recipe']['assetops_commit'] = subprocess.run(
        ['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    sys.path.insert(0, str(Path(root) / 'src'))
    try:
        # Patch an isolated copy so the shared checkout can stay unmodified.
        target = tmp_path / relative
        target.parent.mkdir(parents=True)
        shutil.copyfile(Path(root) / relative, target)
        if 'conformal_factory' not in target.read_text():  # Accept an already patched checkout.
            patch = resources.files('vtsfm.recipes').joinpath('assetops-conformal-factory.patch').read_text()
            result = subprocess.run(['patch', '--batch', '-p1'], input=patch,
                                    cwd=tmp_path, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
        importlib.import_module('servers.tsfm.substrate.resolver')
        spec = importlib.util.spec_from_file_location('servers.tsfm.engine.patched_composition', target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(Path(root) / 'src'))


def _run(composition, recipe, key, **options):
    start = time.perf_counter()
    entry = RECORD['recipe'][key] = {}
    try:
        output = composition.run_recipe(None, load_series(), recipe, **options)
        entry.update(n_anomalies=output['n_anomalies'], n_observations=output['n_observations'],
                     anomaly_indices=np.flatnonzero(output['labels']).tolist())
        return output
    except BaseException as error:
        entry['error'] = f'{type(error).__name__}: {error}'[:2000]
        raise
    finally:
        # Includes the first model load in this process.
        entry['wall_s'] = round(time.perf_counter() - start, 3)


def test_capped_chiller6_recipe_labels(composition):
    from vtsfm.recipes.conformal import BatchedConformalIntervals

    recipe = {'task': 'tsfm_anomaly_detection', 'method': 'conformal', 'estimator': card(),
              'fh': list(range(1, HORIZON + 1)), 'conformal': {'coverage': 0.9}}
    callback = None
    if ADAPTER == 'toto2':
        from vtsfm.recipes.toto import TotoCalibrationBatcher
        callback = TotoCalibrationBatcher()
    RECORD['recipe']['capped_batch_callback'] = type(callback).__name__ if callback else None
    factory = partial(BatchedConformalIntervals, cap_calibration_horizon=True, batch_predict=callback)
    capped = _run(composition, recipe, 'capped', conformal_factory=factory)
    print(f'{ADAPTER} capped recipe: {RECORD["recipe"]["capped"]}')
    assert capped['n_observations'] == HORIZON + 2048
    # The uncapped recipe asks TimesFM for 1,844 steps (above its 128-point head)
    # and ran Moirai out of memory on a 40 GB A100, so it is compared where feasible.
    if ADAPTER == 'timesfm2' or (ADAPTER == 'moirai2' and os.environ.get('VTSFM_RECIPE_COMPARE_UNCAPPED') != '1'):
        RECORD['recipe']['uncapped'] = {'skipped': 'not feasible or not requested'}
        pytest.skip(f'Uncapped {ADAPTER} recipe is not run; capped labels recorded')
    uncapped = _run(composition, recipe, 'uncapped')
    RECORD['recipe']['labels_equal'] = capped['labels'] == uncapped['labels']
    assert capped['labels'] == uncapped['labels']
