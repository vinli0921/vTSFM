"""Manual observed-label forecast quality checks against native and seasonal naive."""

import pytest

from benchmarks.model_evaluation import score_forecasts, seasonal_naive_forecasts
from model_integration_support import QUANTILES, compare_response
from model_serving_support import record_check, save_json


pytestmark = pytest.mark.gpu


def test_observed_forecast_quality_preserves_native(served_model):
    run = served_model
    scores = []
    for index, entry in enumerate(run['cases']):
        case = entry['case']
        check = {'case': case['name'], 'status': 'failed'}
        try:
            payload = {'model': run['spec'].model_id, 'task': 'plugin', 'data': case['data']}
            save_json(run['output'] / f'{index}-request.json', payload)
            response = run['session'].post(run['url'] + '/pooling', json=payload,
                                           timeout=run['parity_timeout_s'])
            (run['output'] / f'{index}-response.txt').write_text(response.text)
            response.raise_for_status()
            body = response.json()
            check.update(compare_response(body, case, entry['expected']))
            values = [[prediction[str(q)] for q in QUANTILES]
                      for prediction in body['data']['predictions']]
            native = score_forecasts(case, entry['expected'])
            served = score_forecasts(case, values)
            naive = score_forecasts(case, seasonal_naive_forecasts(case))
            assert served['counts']['observed_targets'] > 0, 'No observed labels were scored'
            delta = {name: (served[name] - native[name] if native[name] is not None else None)
                     for name in ('MAE', 'MASE', 'SQL')}
            result = {'case': case['name'], 'native': native, 'vtsfm': served,
                      'seasonal_naive': naive, 'served_minus_native': delta}
            scores.append(result)
            save_json(run['output'] / 'accuracy.json', {
                'point_forecast': 'median (0.5 quantile)', 'quantile_levels': list(QUANTILES),
                'gate': 'native numerical parity; no absolute model-accuracy threshold', 'cases': scores})
            check.update(status='passed', accuracy=result)
        except BaseException as error:
            check['error'] = repr(error)
            raise
        finally:
            record_check(run, check)
