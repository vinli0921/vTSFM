"""CPU regressions for the manual GPU HTTP parity check's assertions."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _forecast_fixture():
    case = {
        'name': 'synthetic-distinct-series',
        'data': {
            'inputs': [{'item_id': 'first', 'target': [1, 2]},
                       {'item_id': 'second', 'target': [3, 4, 5]}],
            'parameters': {'prediction_length': 3, 'quantile_levels': [0.2, 0.8]},
        },
    }
    expected = [[[100 * series + quantile + step / 10 for step in range(3)]
                 for quantile in range(9)] for series in range(2)]
    response = {'data': {'predictions': [
        {'item_id': 'first', '0.2': [1, 1.1, 1.2], '0.8': [7, 7.1, 7.2]},
        {'item_id': 'second', '0.2': [101, 101.1, 101.2], '0.8': [107, 107.1, 107.2]},
    ]}}
    return case, expected, response


def test_support_import_does_not_load_model_or_array_packages():
    directory = str(Path(__file__).parent)
    result = subprocess.run([sys.executable, '-c',
        'import sys; sys.path.insert(0, sys.argv[1]); import model_integration_support; '
        'assert not {"torch", "vllm", "numpy", "timesfm", "toto2", "uni2ts"} '
        '& set(sys.modules)', directory], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('adapter,horizon', [('timesfm25', 256), ('toto2', 800), ('moirai2', 160)])
def test_synthetic_cases_cover_ragged_block_decode_and_repeat(adapter, horizon):
    from model_integration_support import integration_cases
    from vtsfm.adapters.forecast import parse_request

    cases = integration_cases(adapter)
    settings = importlib.import_module(f'vtsfm.adapters.{adapter}.spec').FORECAST_SETTINGS
    for case in cases:
        assert 'synthetic' in case['name']
        parse_request(case['data'], settings)
    assert len(cases) == 4
    assert len({case['name'] for case in cases}) == 4
    assert [len(row['target']) for row in cases[1]['data']['inputs']] == [97, 163]
    assert cases[1]['data']['parameters'] == {'prediction_length': 33, 'quantile_levels': [0.2, 0.8]}
    assert cases[2]['data']['parameters']['prediction_length'] == horizon
    assert cases[0]['data'] == cases[-1]['data']
    assert cases == integration_cases(adapter)


def test_response_comparison_selects_sparse_canonical_quantiles_and_reports_errors():
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    response['data']['predictions'][0]['0.2'][0] += 0.00001
    stats = compare_response(response, case, expected)
    assert stats['compared_values'] == 12
    assert stats['max_abs_error'] == pytest.approx(0.00001)
    assert stats['max_rel_error'] == pytest.approx(0.00001)
    json.dumps(stats, allow_nan=False)


def test_response_comparison_rejects_misrouted_item_ids():
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    response['data']['predictions'].reverse()
    with pytest.raises(AssertionError, match='synthetic-distinct-series.*item_id'):
        compare_response(response, case, expected)


def test_response_comparison_rejects_wrong_quantile_mapping():
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    prediction = response['data']['predictions'][0]
    prediction['0.2'], prediction['0.8'] = prediction['0.8'], prediction['0.2']
    with pytest.raises(AssertionError, match='quantile 0.2.*step 0.*expected'):
        compare_response(response, case, expected)


@pytest.mark.parametrize('change', ['missing', 'extra', 'mean'])
def test_response_comparison_rejects_wrong_quantile_keys(change):
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    prediction = response['data']['predictions'][0]
    if change == 'missing':
        del prediction['0.8']
    else:
        prediction['mean' if change == 'mean' else '0.5'] = [0, 0, 0]
    with pytest.raises(AssertionError, match='keys'):
        compare_response(response, case, expected)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_response_comparison_rejects_nonfinite_or_nonnumeric_values(value):
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    response['data']['predictions'][0]['0.2'][0] = value
    with pytest.raises(AssertionError, match='finite number'):
        compare_response(response, case, expected)


@pytest.mark.parametrize('change', ['series', 'quantiles', 'horizon', 'response-horizon'])
def test_response_comparison_rejects_wrong_shapes(change):
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    if change == 'series':
        expected.pop()
    elif change == 'quantiles':
        expected[0].pop()
    elif change == 'horizon':
        expected[0][0].pop()
    else:
        response['data']['predictions'][0]['0.2'].pop()
    with pytest.raises(AssertionError, match='shape|horizon'):
        compare_response(response, case, expected)


def test_response_comparison_rejects_unrequested_native_nonfinite_channels():
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    expected[0][0][0] = float('nan')
    with pytest.raises(AssertionError, match='native.*finite number'):
        compare_response(response, case, expected)


def test_response_comparison_reports_numeric_parity_failure():
    from model_integration_support import compare_response

    case, expected, response = _forecast_fixture()
    response = deepcopy(response)
    response['data']['predictions'][1]['0.8'][2] += 0.1
    with pytest.raises(AssertionError, match='series 1.*quantile 0.8.*step 2.*expected.*tolerance'):
        compare_response(response, case, expected)
