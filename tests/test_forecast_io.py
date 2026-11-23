import base64
from dataclasses import replace

import numpy as np
import pytest

from vtsfm.adapters.forecast import encode_inputs_b64, format_predictions, parse_request
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


# Compact inputs_b64 form (#73): same staged values and the same error contract.

def _json_and_b64(targets, item_ids=None, parameters=None):
    inputs = [{'target': list(target)} for target in targets]
    if item_ids is not None:
        for item, item_id in zip(inputs, item_ids):
            item['item_id'] = item_id
    extra = {} if parameters is None else {'parameters': parameters}
    return ({'inputs': inputs, **extra},
            {'inputs_b64': encode_inputs_b64(targets, item_ids), **extra})


def _b64(values, shape=None, **fields):
    array = np.asarray(values, dtype='<f4')
    encoded = {'data': base64.b64encode(array.tobytes()).decode('ascii'),
               'shape': list(array.shape) if shape is None else shape, **fields}
    return {'inputs_b64': encoded}


@pytest.mark.parametrize('rows,width', [(1, 1), (1, 128), (3, 97), (16, SETTINGS.max_context)])
@pytest.mark.parametrize('with_ids', [False, True])
def test_b64_inputs_parse_identically_to_json_lists(rows, width, with_ids):
    rng = np.random.default_rng(rows * width)
    # Python floats and ints, as JSON decoding produces, including FP32 extremes.
    targets = (rng.standard_normal((rows, width)) * 1e4).tolist()
    targets[0][0] = 3.402823466e38
    targets[-1][-1] = 7
    item_ids = [f'series-{row}' for row in range(rows)] if with_ids else None
    parameters = {'prediction_length': 33, 'quantile_levels': [0.1, 0.5, 0.9]}
    json_data, b64_data = _json_and_b64(targets, item_ids, parameters)
    expected, horizon, levels = parse_request(json_data, SETTINGS)
    observed, b64_horizon, b64_levels = parse_request(b64_data, SETTINGS)
    assert (b64_horizon, b64_levels) == (horizon, levels)
    assert [set(item) for item in observed] == [set(item) for item in expected]
    for got, want in zip(observed, expected, strict=True):
        assert got.get('item_id') == want.get('item_id')
        assert got['target'].dtype == np.float32 and got['target'].shape == (width,)
        np.testing.assert_array_equal(got['target'], want['target'])


def test_b64_defaults_match_json_and_the_request_is_not_modified():
    json_data, b64_data = _json_and_b64([[1, 2.5, 3]])
    original = dict(b64_data['inputs_b64'])
    _, horizon, levels = parse_request(b64_data, SETTINGS)
    assert (horizon, levels) == parse_request(json_data, SETTINGS)[1:]
    assert b64_data['inputs_b64'] == original


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_b64_rejects_nonfinite_values_like_json_lists(value):
    with pytest.raises(ValueError, match='target must be a finite univariate FP32 series'):
        parse_request(_b64([[1.0, value, 3.0]]), SETTINGS)


def test_b64_rejects_empty_and_overlong_series_like_json_lists():
    message = f'target must contain 1 to {SETTINGS.max_context} observations'
    with pytest.raises(ValueError, match=message):
        parse_request(_b64(np.zeros((1, 0))), SETTINGS)
    with pytest.raises(ValueError, match='target must contain 1 to 2 observations'):
        parse_request(_b64([[1, 2, 3]]), replace(SETTINGS, max_context=2))


@pytest.mark.parametrize('change', [
    {'shape': [2, 3]}, {'shape': [1, 2]}, {'shape': [3, 2]},  # Wrong length for the payload.
    {'shape': [0, 3]}, {'shape': [1]}, {'shape': [1, 3, 1]}, {'shape': '1,3'},
    {'shape': [1.0, 3]}, {'shape': [True, 3]}, {'shape': [1, -3]},
    {'data': 'AAAAAAAAAAAAAAAA'[:-1]},  # Truncated.
    {'data': 'AAAA*AAAAAAAAAAA'},  # Outside the base64 alphabet.
    {'data': 'AAAAAAAAAAAAAAAA\n'}, {'data': 'ÀAAAAAAAAAAAAAAA'}, {'data': None}, {'data': [1, 2, 3]},
    {'item_ids': ['a', 'b']}, {'item_ids': 'a'}, {'item_ids': [1]}, {'item_ids': [None]},
])
def test_b64_rejects_malformed_payloads(change):
    data = _b64([[1, 2, 3]])
    data['inputs_b64'].update(change)
    with pytest.raises(ValueError):
        parse_request(data, SETTINGS)


def test_b64_rejects_data_of_another_dtype_size():
    # The same 24 bytes are six FP32 values, not three float64 values of shape [1, 3].
    raw = np.asarray([[1, 2, 3]], dtype='<f8').tobytes()
    data = {'inputs_b64': {'data': base64.b64encode(raw).decode(), 'shape': [1, 3]}}
    with pytest.raises(ValueError, match='inputs_b64.data'):
        parse_request(data, SETTINGS)


@pytest.mark.parametrize('data', [
    {'inputs': [{'target': [1]}], **_b64([[1]])},  # Both forms.
    {'parameters': {}},  # Neither form.
    {'inputs_b64': {'data': 'AACAPw=='}},  # No shape.
    {'inputs_b64': {'shape': [1, 1]}},  # No data.
    {'inputs_b64': {**_b64([[1]])['inputs_b64'], 'sizes': [1]}},  # Ragged layouts stay on JSON lists.
    {'inputs_b64': [[1.0]]},
    {**_b64([[1]]), 'covariates': {}},
])
def test_b64_request_structure_is_validated(data):
    with pytest.raises(ValueError):
        parse_request(data, SETTINGS)


def test_b64_still_validates_parameters():
    data = {**_b64([[1, 2]]), 'parameters': {'prediction_length': SETTINGS.max_horizon + 1}}
    with pytest.raises(ValueError, match='prediction_length'):
        parse_request(data, SETTINGS)


def test_b64_client_helper_rejects_ragged_series():
    with pytest.raises(ValueError, match='equal-length'):
        encode_inputs_b64([[1, 2], [3]])
