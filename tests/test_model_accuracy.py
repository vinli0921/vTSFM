"""Manual observed-label forecast quality checks against native and seasonal naive."""

import pytest

from benchmarks.model_evaluation import score_forecasts, seasonal_naive_forecasts
from model_integration_support import QUANTILES, compare_response
from model_serving_support import record_check, save_json


pytestmark = pytest.mark.gpu


def test_observed_forecast_quality_preserves_native(served_model):
    """Score every case, then fail if any case broke native parity or errored.

    A parity or request failure on one case is recorded with its scores (when
    the response was usable) and the run continues, so one case cannot hide
    every later case's results. The gate is unchanged: any failure fails the test.
    """
    run = served_model
    scores, failures = [], []
    for index, entry in enumerate(run['cases']):
        case = entry['case']
        check = {'case': case['name'], 'status': 'failed'}
        try:
            payload = {'model': run['spec'].model_id, 'task': 'plugin', 'data': case['data']}
            save_json(run['output'] / f'{index}-request.json', payload)
            response = run['session'].post(run['url'] + '/pooling', json=payload, timeout=180)
            (run['output'] / f'{index}-response.txt').write_text(response.text)
            response.raise_for_status()
            body = response.json()
            try:
                parity, parity_error = compare_response(body, case, entry['expected']), None
            except AssertionError as error:
                parity, parity_error = {}, error
            values = [[prediction[str(q)] for q in QUANTILES]
                      for prediction in body['data']['predictions']]
            native = score_forecasts(case, entry['expected'])
            served = score_forecasts(case, values)
            naive = score_forecasts(case, seasonal_naive_forecasts(case))
            assert served['counts']['observed_targets'] > 0, 'No observed labels were scored'
            delta = {name: (served[name] - native[name] if native[name] is not None else None)
                     for name in ('MAE', 'MASE', 'SQL')}
            result = {'case': case['name'], 'parity': 'failed' if parity_error else 'passed',
                      'native': native, 'vtsfm': served, 'seasonal_naive': naive,
                      'served_minus_native': delta}
            if parity_error:
                result['parity_error'] = str(parity_error)
            scores.append(result)
            save_json(run['output'] / 'accuracy.json', {
                'point_forecast': 'median (0.5 quantile)', 'quantile_levels': list(QUANTILES),
                'gate': 'native numerical parity; no absolute model-accuracy threshold', 'cases': scores})
            check.update(parity, accuracy=result)
            if parity_error:
                raise parity_error
            check['status'] = 'passed'
        except Exception as error:
            check['error'] = repr(error)
            failures.append(f"{case['name']}: {error}")
        except BaseException as error:
            check['error'] = repr(error)
            raise
        finally:
            record_check(run, check)
    if failures:
        pytest.fail(f'{len(failures)} of {len(run["cases"])} accuracy cases failed:\n' + '\n'.join(failures),
                    pytrace=False)
