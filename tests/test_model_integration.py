"""Real CUDA/HTTP parity checks, including pinned AssetOpsBench observations."""

import pytest

from model_integration_support import compare_response
from model_serving_support import INTEGRATION_CASES, record_check, save_json


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('index', range(INTEGRATION_CASES),
                         ids=['single', 'ragged', 'long-horizon', 'repeat', 'assetops-1', 'assetops-2'])
def test_http_forecast_matches_native(served_model, index):
    run = served_model
    entry = run['cases'][index]
    case = entry['case']
    payload = {'model': run['spec'].model_id, 'task': 'plugin', 'data': case['data']}
    record = {'case': case['name'], 'status': 'failed'}
    try:
        save_json(run['output'] / f'{index}-request.json', payload)
        response = run['session'].post(run['url'] + '/pooling', json=payload,
                                       timeout=run['parity_timeout_s'])
        (run['output'] / f'{index}-response.txt').write_text(response.text)
        response.raise_for_status()
        body = response.json()
        request_id = body.get('request_id')
        assert isinstance(request_id, str) and request_id, 'Missing HTTP request ID'
        assert request_id not in run['request_ids'], 'Server reused an earlier HTTP request ID'
        run['request_ids'].add(request_id)
        record.update(compare_response(body, case, entry['expected']))
        record['status'] = 'passed'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record_check(run, record)
