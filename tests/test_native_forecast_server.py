"""CPU checks for the manual native HTTP baseline; no model downloads."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from vtsfm.adapters import get_adapter


class FakeForecaster:
    """Replace only GPU inference while keeping request and response IO real."""

    def __init__(self, adapter):
        self.adapter = adapter

    def predict(self, data):
        return self.predict_array(data).tolist()

    def predict_array(self, data):
        import numpy as np

        horizon = data['parameters']['prediction_length']
        return np.asarray([[[float(item['target'][0] + channel + step / 10)
                             for step in range(horizon)] for channel in range(9)]
                           for item in data['inputs']])

    def metadata(self):
        spec = get_adapter(self.adapter)
        return {'adapter': self.adapter, 'model_id': spec.model_id,
                'model_revision': spec.model_revision, 'model_path': '/fake/checkpoint',
                'runtime': {'uuid': 'GPU-native-test', 'dtype': 'float32'}}


def _data():
    return {'inputs': [{'item_id': 'first', 'target': [10, 11]},
                       {'target': [20, 21, 22]}],
            'parameters': {'prediction_length': 2, 'quantile_levels': [0.2, 0.8]}}


async def _request(app, method, path, payload=None):
    """Minimal ASGI client: the CPU environment does not require httpx."""
    content = b'' if payload is None else json.dumps(payload).encode()
    messages = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {'type': 'http.request', 'body': content, 'more_body': False}
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)

    await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
               'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
               'query_string': b'', 'root_path': '',
               'headers': [(b'content-type', b'application/json')],
               'client': ('127.0.0.1', 12345), 'server': ('127.0.0.1', 8000)}, receive, send)
    status = next(message['status'] for message in messages if message['type'] == 'http.response.start')
    body = b''.join(message.get('body', b'') for message in messages
                    if message['type'] == 'http.response.body')
    return status, json.loads(body)


@pytest.fixture
def native_app(monkeypatch):
    pytest.importorskip('fastapi')
    import native_forecast_server

    monkeypatch.setenv('VTSFM_TEST_ADAPTER', 'timesfm25')
    monkeypatch.setattr(native_forecast_server, 'NativeForecaster', FakeForecaster)
    return native_forecast_server.create_app()


def test_server_import_and_reference_help_do_not_import_model_packages():
    directory = str(Path(__file__).parent)
    result = subprocess.run([sys.executable, '-c',
        'import sys; sys.path.insert(0, sys.argv[1]); import native_forecast_server; '
        'assert not {"torch", "vllm", "numpy", "timesfm", "toto2", "uni2ts", "fastapi"} '
        '& set(sys.modules)', directory], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(Path(directory) / 'model_integration_support.py'),
                             '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--cases' in result.stdout


def test_health_and_metadata_require_successful_model_startup(native_app):
    async def scenario():
        assert (await _request(native_app, 'GET', '/health'))[0] == 503
        assert (await _request(native_app, 'GET', '/metadata'))[0] == 503
        async with native_app.router.lifespan_context(native_app):
            assert await _request(native_app, 'GET', '/health') == (200, {'status': 'ok'})
            status, metadata = await _request(native_app, 'GET', '/metadata')
            assert status == 200
            assert metadata['runtime']['uuid'] == 'GPU-native-test'
        assert (await _request(native_app, 'GET', '/health'))[0] == 503

    asyncio.run(scenario())


def test_http_routes_sparse_quantiles_ids_and_unique_request_ids(native_app):
    async def scenario():
        payload = {'model': get_adapter('timesfm25').model_id, 'task': 'plugin', 'data': _data()}
        async with native_app.router.lifespan_context(native_app):
            responses = [await _request(native_app, 'POST', '/pooling', payload) for _ in range(2)]
        for status, response in responses:
            assert status == 200
            assert type(response['created_at']) is int
            assert response['data']['predictions'] == [
                {'item_id': 'first', '0.2': [11.0, 11.1], '0.8': [17.0, 17.1]},
                {'0.2': [21.0, 21.1], '0.8': [27.0, 27.1]},
            ]
        assert responses[0][1]['request_id'] != responses[1][1]['request_id']

    asyncio.run(scenario())


def test_http_formats_native_fp32_arrays_without_using_json_forecasts(native_app, monkeypatch):
    import numpy as np
    import native_forecast_server

    class ArrayForecaster(FakeForecaster):
        def predict(self, data):
            raise AssertionError('HTTP must not materialize the complete JSON forecast')

        def predict_array(self, data):
            values = super().predict_array(data).astype(np.float32)
            # Exactly representable FP32 values make the HTTP expectation literal.
            values[:, :, 1] = values[:, :, 0] + 0.125
            return values

    monkeypatch.setattr(native_forecast_server, 'NativeForecaster', ArrayForecaster)

    async def scenario():
        async with native_app.router.lifespan_context(native_app):
            status, body = await _request(native_app, 'POST', '/pooling', {
                'model': get_adapter('timesfm25').model_id, 'task': 'plugin', 'data': _data()})
        assert status == 200
        assert body['data']['predictions'] == [
            {'item_id': 'first', '0.2': [11, 11.125], '0.8': [17, 17.125]},
            {'0.2': [21, 21.125], '0.8': [27, 27.125]},
        ]

    asyncio.run(scenario())


def test_native_json_forecast_wrapper_preserves_finite_validation(monkeypatch):
    import numpy as np
    from model_integration_support import NativeForecaster, forecast_settings

    forecaster = object.__new__(NativeForecaster)
    forecaster.settings = forecast_settings('timesfm25')
    values = np.zeros((2, 9, 2), dtype=np.float32)
    values[0, 1] = [1, 1.125]
    monkeypatch.setattr(forecaster, 'predict_array', lambda data: values)
    forecast = forecaster.predict(_data())
    assert isinstance(forecast, list)
    assert forecast[0][1] == [1, 1.125]
    json.dumps(forecast, allow_nan=False)
    values[0, 0, 0] = np.nan
    with pytest.raises(AssertionError, match='finite number'):
        forecaster.predict(_data())


@pytest.mark.parametrize('change', [
    'wrong-model', 'wrong-task', 'extra-envelope-field', 'missing-data',
    'zero-horizon', 'overlong-horizon', 'boolean-horizon', 'numeric-id',
    'unsupported-quantile', 'unsorted-quantiles', 'duplicate-quantiles', 'covariates',
])
def test_http_rejects_unsupported_forecast_requests(native_app, change):
    payload = {'model': get_adapter('timesfm25').model_id, 'task': 'plugin', 'data': _data()}
    if change == 'wrong-model':
        payload['model'] = 'other/model'
    elif change == 'wrong-task':
        payload['task'] = 'embed'
    elif change == 'extra-envelope-field':
        payload['extra'] = True
    elif change == 'missing-data':
        del payload['data']
    elif change == 'numeric-id':
        payload['data']['inputs'][0]['item_id'] = 42
    elif change == 'covariates':
        payload['data']['inputs'][0]['past_covariates'] = {'x': [1, 2]}
    elif change.endswith('horizon'):
        payload['data']['parameters']['prediction_length'] = {
            'zero-horizon': 0, 'overlong-horizon': 1025, 'boolean-horizon': True}[change]
    else:
        payload['data']['parameters']['quantile_levels'] = {
            'unsupported-quantile': [0.05], 'unsorted-quantiles': [0.8, 0.2],
            'duplicate-quantiles': [0.2, 0.2]}[change]

    async def scenario():
        async with native_app.router.lifespan_context(native_app):
            status, _ = await _request(native_app, 'POST', '/pooling', payload)
            assert status == 422

    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['nonfinite', 'quantile-count', 'horizon'])
def test_http_rejects_invalid_native_output(native_app, monkeypatch, change):
    import native_forecast_server

    class InvalidForecaster(FakeForecaster):
        def predict_array(self, data):
            values = super().predict_array(data)
            if change == 'nonfinite':
                values[0][1][0] = float('nan')
            elif change == 'quantile-count':
                values = values[:, :-1]
            else:
                values = values[:, :, :-1]
            return values

    monkeypatch.setattr(native_forecast_server, 'NativeForecaster', InvalidForecaster)

    async def scenario():
        async with native_app.router.lifespan_context(native_app):
            status, response = await _request(native_app, 'POST', '/pooling', {
                'model': get_adapter('timesfm25').model_id, 'task': 'plugin', 'data': _data()})
            assert status == 500
            assert ('nonfinite' if change == 'nonfinite' else 'shape') in response['detail']

    asyncio.run(scenario())


def test_concurrent_http_requests_execute_one_forecast_at_a_time(native_app, monkeypatch):
    import native_forecast_server

    class ConcurrentForecaster(FakeForecaster):
        active = maximum = 0
        guard = threading.Lock()

        def predict_array(self, data):
            with self.guard:
                type(self).active += 1
                type(self).maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.02)
                return super().predict_array(data)
            finally:
                with self.guard:
                    type(self).active -= 1

    monkeypatch.setattr(native_forecast_server, 'NativeForecaster', ConcurrentForecaster)

    async def scenario():
        async with native_app.router.lifespan_context(native_app):
            payloads = []
            for index in range(6):
                data = _data()
                data['inputs'][0]['target'][0] = index
                payloads.append({'model': get_adapter('timesfm25').model_id,
                                 'task': 'plugin', 'data': data})
            responses = await asyncio.gather(*[
                _request(native_app, 'POST', '/pooling', payload) for payload in payloads])
        assert [body['data']['predictions'][0]['0.2'][0] for _, body in responses] == [1, 2, 3, 4, 5, 6]
        assert all(status == 200 for status, _ in responses)
        assert len({body['request_id'] for _, body in responses}) == 6
        assert ConcurrentForecaster.maximum == 1

    asyncio.run(scenario())


def test_native_reference_accepts_custom_cases_and_defaults_without_mutating(monkeypatch):
    import model_integration_support as support

    monkeypatch.setattr(support, 'NativeForecaster', FakeForecaster)
    cases = [{'name': 'observed-window', 'data': _data()}]
    original = deepcopy(cases)
    result = support.native_reference('timesfm25', cases)
    assert cases == original
    assert result['cases'][0]['case'] == original[0]
    assert result['cases'][0]['expected'][0][1] == [11, 11.1]
    assert result['runtime']['uuid'] == 'GPU-native-test'
    result = support.native_reference('timesfm25')
    assert len(result['cases']) == 4


def test_native_reference_rejects_invalid_cases_before_loading_gpu(monkeypatch):
    import model_integration_support as support

    def unexpected_load(adapter):
        pytest.fail('Invalid cases must fail before loading the model')

    monkeypatch.setattr(support, 'NativeForecaster', unexpected_load)
    invalid = _data()
    invalid['parameters']['prediction_length'] = 1025
    with pytest.raises(ValueError, match='prediction_length'):
        support.native_reference('timesfm25', [{'name': 'invalid', 'data': invalid}])


def test_native_reference_cli_loads_case_manifest_and_preserves_provenance(tmp_path, monkeypatch):
    import model_integration_support as support

    monkeypatch.setattr(support, 'NativeForecaster', FakeForecaster)
    manifest = {'cases': [{'name': 'observed-window', 'data': _data()}],
                'provenance': {'source': 'cpu-fixture'}}
    source = tmp_path / 'cases.json'
    output = tmp_path / 'native.json'
    source.write_text(json.dumps(manifest))
    assert support.main(['--adapter', 'timesfm25', '--cases', str(source), '--output', str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved['provenance'] == {'source': 'cpu-fixture'}
    assert saved['cases'][0]['case']['name'] == 'observed-window'


@pytest.mark.parametrize('manifest', [None, {'cases': None}, {}, 'not-cases'])
def test_native_reference_cli_does_not_replace_invalid_manifests_with_synthetic_cases(
        tmp_path, monkeypatch, manifest):
    import model_integration_support as support

    def unexpected_load(adapter):
        pytest.fail('Invalid manifests must fail before loading the model')

    monkeypatch.setattr(support, 'NativeForecaster', unexpected_load)
    source = tmp_path / 'cases.json'
    source.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        support.main(['--adapter', 'timesfm25', '--cases', str(source),
                      '--output', str(tmp_path / 'native.json')])
def test_native_cases_preserve_real_data_metadata_without_model_imports():
    from model_integration_support import integration_cases, validate_cases

    case = integration_cases('timesfm25')[0]
    case.update(evaluation={'targets': [[1.0]], 'observed': [[True]], 'seasonality': 24},
                provenance={'kind': 'observed fixture'})
    validated = validate_cases('timesfm25', [case])
    assert validated == [case]
    case['provenance']['kind'] = 'changed'
    assert validated[0]['provenance']['kind'] == 'observed fixture'

