"""Opt-in CPU checks against actual Chronos execution, with tiny random weights.

No checkpoint is downloaded. These establish native API/patch-schedule behavior,
not production-checkpoint accuracy, GPU parity or a measured performance result.
"""

import os

import numpy as np
import pytest

if os.environ.get('VTSFM_NATIVE_CHRONOS_TEST') != '1':
    pytest.skip('Set VTSFM_NATIVE_CHRONOS_TEST=1 in a Chronos + sktime environment', allow_module_level=True)

torch = pytest.importorskip('torch')
pd = pytest.importorskip('pandas')
Chronos2CoreConfig = pytest.importorskip('chronos.chronos2.config').Chronos2CoreConfig
Chronos2Model = pytest.importorskip('chronos.chronos2.model').Chronos2Model
Chronos2Pipeline = pytest.importorskip('chronos.chronos2.pipeline').Chronos2Pipeline
Chronos2Forecaster = pytest.importorskip('sktime.forecasting.chronos2').Chronos2Forecaster
ConformalIntervals = pytest.importorskip('sktime.forecasting.conformal').ConformalIntervals
BatchedConformalIntervals = pytest.importorskip('vtsfm.recipes.conformal').BatchedConformalIntervals
ChronosCalibrationBatcher = pytest.importorskip('vtsfm.recipes.chronos').ChronosCalibrationBatcher


@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
@pytest.mark.parametrize('context_cap', [None, 12])
def test_real_native_patch_batches_match_serial_residuals_and_intervals(monkeypatch, attention, context_cap):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        config = Chronos2CoreConfig(d_model=16, d_ff=32, d_kv=4, num_heads=4,
            num_layers=1, dropout_rate=0, attn_implementation=attention,
            chronos_config={'context_length': 64, 'input_patch_size': 4,
                'output_patch_size': 4, 'input_patch_stride': 4,
                'quantiles': [i / 10 for i in range(1, 10)], 'max_output_patches': 4,
                'use_reg_token': True})
        pipeline = Chronos2Pipeline(Chronos2Model(config).eval())
        monkeypatch.setattr(Chronos2Forecaster, '_load_pipeline', lambda self: pipeline)
        native_predict = pipeline.predict
        calls = []

        def predict(inputs, **options):
            calls.append((len(inputs), options['prediction_length']))
            return native_predict(inputs, **options)

        monkeypatch.setattr(pipeline, 'predict', predict)
        forecaster = Chronos2Forecaster(config={'context_length': context_cap}, seed=42)
        y = pd.Series(np.sin(np.arange(30) / 3), name='sensor', index=pd.RangeIndex(30, name='time'))
        original = ConformalIntervals(forecaster, initial_window=10)
        original.fit(y, fh=[1, 2, 3])
        serial_calls = len(calls)
        calls.clear()
        candidate = BatchedConformalIntervals(forecaster, initial_window=10,
            batch_predict=ChronosCalibrationBatcher(), batch_size=64)
        candidate.fit(y, fh=[1, 2, 3])
        assert len(calls) < serial_calls
        assert any(rows > 1 for rows, _ in calls)
        pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_,
                                      atol=1e-5, rtol=1e-5)
        pd.testing.assert_frame_equal(candidate.predict_interval(coverage=[0.5, 0.9]),
            original.predict_interval(coverage=[0.5, 0.9]), atol=1e-5, rtol=1e-5)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.skipif(os.environ.get('VTSFM_CHRONOS_ORIGINAL_SHAPE_TEST') != '1',
                    reason='Set VTSFM_CHRONOS_ORIGINAL_SHAPE_TEST=1 for the 2048-point native CPU check')
def test_original_2048_point_shape_batches_unchanged_growing_contexts(monkeypatch, request):
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = Chronos2CoreConfig(d_model=8, d_ff=16, d_kv=4, num_heads=2,
        num_layers=1, dropout_rate=0, attn_implementation='sdpa',
        chronos_config={'context_length': 8192, 'input_patch_size': 16,
            'output_patch_size': 16, 'input_patch_stride': 16,
            'quantiles': [i / 10 for i in range(1, 10)], 'max_output_patches': 64,
            'use_reg_token': True})
    pipeline = Chronos2Pipeline(Chronos2Model(config).eval())
    monkeypatch.setattr(Chronos2Forecaster, '_load_pipeline', lambda self: pipeline)
    native_predict = pipeline.predict
    calls = []

    def predict(inputs, **options):
        calls.append((len(inputs), options['prediction_length']))
        return native_predict(inputs, **options)

    monkeypatch.setattr(pipeline, 'predict', predict)
    # Original round4 uses 2144 points, a 96-point holdout and no context cap.
    # Its conformal calibration therefore sees all growing prefixes of 2048.
    y = pd.Series(np.sin(np.arange(2048) / 37), name='sensor')
    forecaster = Chronos2Forecaster(seed=0)
    original = ConformalIntervals(forecaster)
    original.fit(y, fh=list(range(1, 97)))
    serial_calls = len(calls)
    calls.clear()
    candidate = BatchedConformalIntervals(forecaster, batch_size=64,
                                         batch_predict=ChronosCalibrationBatcher())
    candidate.fit(y, fh=list(range(1, 97)))
    print(f'original-shape native forecast calls: {serial_calls} serial, {len(calls)} batched')
    assert serial_calls == 1844 and len(calls) < serial_calls / 2
    assert any(rows > 1 and horizon > 1024 for rows, horizon in calls)
    pd.testing.assert_frame_equal(candidate.residuals_matrix_, original.residuals_matrix_, atol=1e-5, rtol=1e-5)
    pd.testing.assert_frame_equal(candidate.predict_interval(coverage=0.9),
                                  original.predict_interval(coverage=0.9), atol=1e-5, rtol=1e-5)
