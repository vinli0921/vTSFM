"""Manual matched native-HTTP/vLLM throughput, latency, and memory experiments."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os

import pytest

from benchmarks.model_performance import measure_requests
from model_integration_support import compare_response, payload_form, wire_data
from model_serving_support import record_check, save_json


pytestmark = pytest.mark.gpu


def test_matched_http_performance(model_run):
    run = model_run
    requests = int(os.environ.get('VTSFM_PERF_REQUESTS', '100'))
    warmups = int(os.environ.get('VTSFM_PERF_WARMUPS', '5'))
    order = os.environ.get('VTSFM_PERF_ORDER', 'native,vtsfm').split(',')
    if sorted(order) != ['native', 'vtsfm']:
        raise ValueError('Backend order must contain native and vtsfm exactly once')
    form = payload_form()
    points = []
    run['metadata']['performance'] = {'requests_per_point': requests, 'warmups': warmups,
                                    'backend_order': order, 'clients': run['clients'],
                                    'payload_form': form}
    for backend in order:
        with run['backend'](backend) as server:
            for entry in run['cases']:
                case = entry['case']
                # Encoding happens here, outside the timed samples; both backends
                # receive identical bytes.
                data, sent_form = wire_data(case['data'], form)
                payload = {'model': run['spec'].model_id, 'task': 'plugin', 'data': data}
                encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
                for clients in run['clients']:
                    check = {'case': case['name'], 'backend': backend, 'clients': clients,
                             'payload_form': sent_form, 'status': 'failed'}
                    try:
                        # Validate a concurrent burst separately; accuracy checks and
                        # payload construction never enter the performance samples.
                        def predict(_):
                            import requests as http
                            with http.Session() as session:
                                session.trust_env = False
                                result = session.post(server['url'] + '/pooling', json=payload,
                                                      timeout=server['parity_timeout_s'])
                                result.raise_for_status()
                                return result.json()
                        pool = ThreadPoolExecutor(max_workers=clients)
                        try:
                            bodies = list(pool.map(predict, range(clients)))
                        except BaseException:
                            server['stop']()
                            raise
                        finally:
                            pool.shutdown(wait=True, cancel_futures=True)
                        parity = [compare_response(body, case, entry['expected']) for body in bodies]
                        assert len({body['request_id'] for body in bodies}) == clients
                        point = measure_requests(server['url'], encoded, len(case['data']['inputs']),
                            requests=requests, clients=clients, warmups=warmups, monitor=run['monitor'],
                            cancel=server['stop'], record_path=run['output'] / 'points' /
                            f'{backend}-{case["name"]}-clients{clients}.json')
                        point.update(backend=backend, case=case['name'],
                                     forecast_execution=run['metadata']['forecast_execution'][backend],
                                     context_lengths=[len(item['target']) for item in case['data']['inputs']],
                                     horizon=case['data']['parameters']['prediction_length'],
                                     payload_form=sent_form, payload_bytes=len(encoded),
                                     payload_sha256=hashlib.sha256(encoded).hexdigest(), parity=parity)
                        points.append(point)
                        save_json(run['output'] / 'performance.json', {'points': points,
                            'source_commit': run['metadata']['commit'], 'precision': run['metadata']['precision'],
                            'forecast_execution': run['metadata']['forecast_execution'],
                            'backend_order': order, 'no_automatic_performance_merge_threshold': True})
                        assert point['status'] == 'passed', f'Measured HTTP requests failed: {point}'
                        check.update(status='passed', performance={key: value for key, value in point.items()
                                     if key not in ('samples', 'nvml_samples', 'parity')})
                    except BaseException as error:
                        check['error'] = repr(error)
                        raise
                    finally:
                        record_check(run, check)
