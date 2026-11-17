from dataclasses import replace

import numpy as np
import pytest

from vtsfm.adapters.forecast import format_predictions, parse_request
from vtsfm.adapters.timesfm25.spec import FORECAST_SETTINGS as SETTINGS


def test_quantiles_keep_series_order_and_do_not_invent_a_mean():
    data = {'inputs': [{'target': [1, 2, 3], 'item_id': 'a'}, {'target': [5, 6], 'item_id': 'b'}],
            'parameters': {'prediction_length': 2, 'quantile_levels': [0.1, 0.5, 0.9]}}
    inputs, horizon, levels = parse_request(data, SETTINGS)
    values = np.arange(36).reshape(2, 9, 2)
    result = format_predictions(values, [item['item_id'] for item in inputs], levels, horizon, SETTINGS)['predictions']
    assert result == [
        {'item_id': 'a', '0.1': [0, 1], '0.5': [8, 9], '0.9': [16, 17]},
        {'item_id': 'b', '0.1': [18, 19], '0.5': [26, 27], '0.9': [34, 35]},
    ]


@pytest.mark.parametrize('target', [
    [], [[1, 2]], [True], [float('nan')], [float('inf')], [1e100], ['1'],
    # NumPy would silently coerce these; validation must still reject them.
    [1.0, True], [1.0, '1.5'], [1.0, None], [1.0, {}], [np.bool_(True)],
    [-float('inf')], [3.5e38], [10**39], [10**400],
])
def test_rejects_unsupported_targets(target):
    with pytest.raises(ValueError, match='target must'):
        parse_request({'inputs': [{'target': target}]}, SETTINGS)


def test_targets_become_fp32_arrays_without_changing_the_request():
    target = [1, 2.5, -3.402823466e38, np.float32(4), np.int64(5)]
    data = {'inputs': [{'target': target, 'item_id': 'a'}]}
    inputs, _, _ = parse_request(data, SETTINGS)
    assert inputs[0]['item_id'] == 'a'
    parsed = inputs[0]['target']
    assert parsed.dtype == np.float32
    np.testing.assert_array_equal(parsed, np.asarray(target, dtype=np.float32))
    assert data['inputs'][0]['target'] is target


@pytest.mark.parametrize('parameters', [
    {'cross_learning': True}, {'prediction_length': True}, {'prediction_length': 1025},
    {'quantile_levels': [0.05]}, {'quantile_levels': [0.9, 0.1]},
])
def test_rejects_unsupported_parameters(parameters):
    with pytest.raises(ValueError):
        parse_request({'inputs': [{'target': [1, 2]}], 'parameters': parameters}, SETTINGS)


def test_rejects_covariates_and_overlong_context():
    with pytest.raises(ValueError):
        parse_request({'inputs': [{'target': [1], 'past_covariates': {'x': [2]}}]}, SETTINGS)
    with pytest.raises(ValueError):
        parse_request({'inputs': [{'target': [1, 2, 3]}]}, replace(SETTINGS, max_context=2))


def test_formats_torch_outputs_and_checks_only_requested_quantiles():
    torch = pytest.importorskip('torch')
    values = torch.arange(18, dtype=torch.float32).reshape(1, 9, 2)
    values[0, 0, 0] = float('nan')  # 0.1 is not requested
    result = format_predictions(values, [None], [0.5, 0.9], 2, SETTINGS)['predictions']
    assert result == [{'0.5': [8.0, 9.0], '0.9': [16.0, 17.0]}]
    with pytest.raises(ValueError, match='nonfinite'):
        format_predictions(values, [None], [0.1, 0.5], 2, SETTINGS)


def test_rejects_incomplete_or_nonfinite_model_outputs():
    for output in (np.zeros((1, 9, 1)), np.full((1, 9, 2), np.nan)):
        with pytest.raises(ValueError):
            format_predictions(output, [None], [0.5], 2, SETTINGS)
