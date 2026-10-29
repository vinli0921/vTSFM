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


@pytest.mark.parametrize('target', [[], [[1, 2]], [True], [float('nan')], [float('inf')], [1e100], ['1']])
def test_rejects_unsupported_targets(target):
    with pytest.raises(ValueError):
        parse_request({'inputs': [{'target': target}]}, SETTINGS)


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


def test_rejects_incomplete_or_nonfinite_model_outputs():
    for output in (np.zeros((1, 9, 1)), np.full((1, 9, 2), np.nan)):
        with pytest.raises(ValueError):
            format_predictions(output, [None], [0.5], 2, SETTINGS)
