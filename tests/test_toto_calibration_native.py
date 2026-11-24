"""Opt-in native Toto CPU checks with tiny random weights and no downloads."""

import os
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

if os.environ.get('VTSFM_NATIVE_TOTO_TEST') != '1':
    pytest.skip('Set VTSFM_NATIVE_TOTO_TEST=1 in a Toto-2 + sktime Python 3.12 environment', allow_module_level=True)

torch = pytest.importorskip('torch')
pd = pytest.importorskip('pandas')
Toto2ModelConfig = pytest.importorskip('toto2.configuration').Toto2ModelConfig
Toto2Model = pytest.importorskip('toto2').Toto2Model
sktime_toto = pytest.importorskip('sktime.forecasting.toto2')
Toto2Forecaster = sktime_toto.Toto2Forecaster
ForecastingHorizon = pytest.importorskip('sktime.forecasting.base').ForecastingHorizon
ConformalIntervals = pytest.importorskip('sktime.forecasting.conformal').ConformalIntervals
BatchedConformalIntervals = pytest.importorskip('vtsfm.recipes.conformal').BatchedConformalIntervals
TotoCalibrationBatcher = pytest.importorskip('vtsfm.recipes.toto').TotoCalibrationBatcher


@pytest.fixture
def model(monkeypatch):
    torch.manual_seed(42)
    config = Toto2ModelConfig(patch_size=4, d_model=16, num_heads=2, num_layers=2,
        layer_group_size=2, num_variate_layers_per_group=1, variate_layer_first=False,
        residual_attn_ratio=1.0)
    model = Toto2Model(config).eval()
    monkeypatch.setattr(sktime_toto, '_CachedToto2Forecaster',
        lambda **kwargs: SimpleNamespace(load_from_checkpoint=lambda: model))
    return model


@pytest.mark.parametrize('decode_block_size', [None, 4, 8])
def test_native_toto_padding_and_decode_schedules_match_serial_calibration(model, monkeypatch, decode_block_size):
    calls = []
    native_forecast = model.forecast

    def forecast(inputs, **options):
        calls.append((inputs['target'].shape[0], options['horizon'], options['has_missing_values']))
        return native_forecast(inputs, **options)

    monkeypatch.setattr(model, 'forecast', forecast)
    forecaster = Toto2Forecaster(device='cpu', seed=42, decode_block_size=decode_block_size)
    y = pd.Series(np.sin(np.arange(30) / 3), name='sensor')
    original = ConformalIntervals(forecaster, initial_window=10)
    original.fit(y, fh=[1, 2, 3])
    serial_calls = len(calls)
    calls.clear()
    candidate = BatchedConformalIntervals(forecaster, initial_window=10,
        batch_predict=TotoCalibrationBatcher(), batch_size=64)
    candidate.fit(y, fh=[1, 2, 3])
    assert len(calls) < serial_calls
    assert any(rows > 1 for rows, _, _ in calls)
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_, atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(candidate.predict_interval(coverage=[0.5, 0.9]),
        original.predict_interval(coverage=[0.5, 0.9]), atol=1e-5, rtol=1e-5)


def test_small_budget_and_sparse_horizons_retain_native_serial_calls(model, monkeypatch):
    calls = []
    native_forecast = model.forecast

    def forecast(inputs, **options):
        calls.append(inputs['target'].shape[0])
        return native_forecast(inputs, **options)

    monkeypatch.setattr(model, 'forecast', forecast)
    forecasters = [Toto2Forecaster(device='cpu', seed=42).fit(
        pd.Series(np.arange(width, dtype=float), name='sensor')) for width in [10, 11]]
    targets = [pd.RangeIndex(10, 14), pd.RangeIndex(11, 14)]
    expected = [fc.predict(ForecastingHorizon(target, is_relative=False))
                for fc, target in zip(forecasters, targets)]
    actual = TotoCalibrationBatcher(max_batch_points=1)(forecasters, targets)
    for left, right in zip(actual, expected):
        pd.testing.assert_series_equal(left, right)
    assert all(rows == 1 for rows in calls)
    calls.clear()
    sparse = [targets[0][::2], targets[1][::2]]
    result = TotoCalibrationBatcher()(forecasters, sparse)
    assert [len(output) for output in result] == [2, 2] and calls == [1, 1]
    calls.clear()
    forecasters = [Toto2Forecaster(device='cpu', seed=42).fit(
        pd.Series(np.arange(width, dtype=float), name='sensor')) for width in [9, 10, 11, 9, 10]]
    targets = [pd.RangeIndex(fc.cutoff[0] + 1, fc.cutoff[0] + 5) for fc in forecasters]
    result = TotoCalibrationBatcher(max_batch_points=100)(forecasters, targets)
    assert calls == [2, 2, 1] and len(result) == 5


def test_cached_model_loading_retains_the_native_seed_order(model, monkeypatch):
    draws = []

    def load():
        draws.append(torch.rand(()).item())
        return model

    monkeypatch.setattr(sktime_toto, '_CachedToto2Forecaster',
                        lambda **kwargs: SimpleNamespace(load_from_checkpoint=load))
    forecasters = [Toto2Forecaster(device='cpu', seed=42).fit(
        pd.Series(np.arange(width, dtype=float), name='sensor')) for width in [10, 11]]
    targets = [pd.RangeIndex(10, 14), pd.RangeIndex(11, 14)]
    torch.manual_seed(777)
    TotoCalibrationBatcher()(forecasters, targets)
    wanted = torch.rand((), generator=torch.Generator().manual_seed(42)).item()
    assert draws == [wanted]


@pytest.mark.parametrize('failure', ['bad-output', 'oom', 'linear-shape', 'unrelated-error'])
def test_toto_batch_failure_retries_only_recoverable_errors(model, monkeypatch, failure):
    sizes = []
    native_forecast = model.forecast
    batch_target = None

    def forecast(inputs, **options):
        nonlocal batch_target
        rows = inputs['target'].shape[0]
        sizes.append(rows)
        if rows == 1:
            assert batch_target() is None
            return native_forecast(inputs, **options)
        batch_target = weakref.ref(inputs['target'])
        if failure == 'oom':
            raise torch.OutOfMemoryError('simulated batch OOM')
        if failure == 'linear-shape':
            torch.nn.functional.linear(torch.zeros(3, 4), torch.zeros(1, 4))
            pytest.fail('Expected the origin shape guard to raise')
        if failure == 'unrelated-error':
            raise ValueError('simulated unrelated model error')
        return native_forecast(inputs, **options)[:, :-1]

    y = pd.Series(np.sin(np.arange(20) / 3), name='sensor')
    forecasters = [Toto2Forecaster(device='cpu', seed=42).fit(y.iloc[:origin])
                   for origin in [10, 11]]
    targets = [y.iloc[origin:14].index for origin in [10, 11]]
    expected = [fc.predict(ForecastingHorizon(index, is_relative=False))
                for fc, index in zip(forecasters, targets)]
    monkeypatch.setattr(model, 'forecast', forecast)
    if failure == 'unrelated-error':
        with pytest.raises(ValueError, match='unrelated model error'):
            TotoCalibrationBatcher()(forecasters, targets)
        assert sizes == [2]
        return
    forecasts = TotoCalibrationBatcher()(forecasters, targets)
    assert sizes == [2, 1, 1]
    for actual, wanted in zip(forecasts, expected):
        pd.testing.assert_series_equal(actual, wanted)


def test_scoped_batch_retains_native_quantiles_and_shared_model_state(model):
    from vtsfm.recipes._toto_linear import _OriginLinearMode

    y = pd.Series(np.sin(np.arange(20) / 3), name='sensor')
    origins = [9, 10, 11]
    forecasters = [Toto2Forecaster(device='cpu', seed=42).fit(y.iloc[:origin])
                   for origin in origins]
    targets = [y.iloc[origin:16].index for origin in origins]
    forwards = {name: module.__dict__.get('forward')
                for name, module in model.named_modules()}
    parameters = {name: (id(value), value.detach().clone())
                  for name, value in model.named_parameters()}
    state_keys = list(model.state_dict())
    saved_inputs = [{key: value.clone() for key, value in fc._inputs_.items()}
                    for fc in forecasters]
    native = [fc._run_forecast(ForecastingHorizon(target, is_relative=False))[1]
              for fc, target in zip(forecasters, targets)]
    actual = TotoCalibrationBatcher()(forecasters, targets)
    aligned = [fc._align_context(fc._inputs_['target'], fc._inputs_['target_mask'], 4)
               for fc in forecasters]
    inputs = {'target': torch.cat([row[0] for row in aligned]),
              'target_mask': torch.cat([row[1] for row in aligned]),
              'series_ids': torch.cat([fc._inputs_['series_ids'] for fc in forecasters])}
    with _OriginLinearMode(len(origins)):
        quantiles = model.forecast(inputs, horizon=7, decode_block_size=None,
                                   has_missing_values=True)
    median = model.output_head.knots.index(0.5)
    for row, (target, expected, prediction) in enumerate(zip(targets, native, actual)):
        assert torch.equal(quantiles[:, row:row+1, ..., :len(target)], expected)
        np.testing.assert_array_equal(prediction.to_numpy(), expected[median, 0, 0].numpy())
    assert list(model.state_dict()) == state_keys
    for name, module in model.named_modules():
        assert module.__dict__.get('forward') is forwards[name]
    for name, value in model.named_parameters():
        assert id(value) == parameters[name][0]
        assert torch.equal(value, parameters[name][1])
    for fc, saved in zip(forecasters, saved_inputs):
        assert all(torch.equal(fc._inputs_[key], value) for key, value in saved.items())
