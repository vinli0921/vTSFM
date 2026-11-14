"""CPU regressions for the manual accuracy check's per-case failure handling."""

import json
from types import SimpleNamespace

import pytest

import test_model_accuracy as accuracy
from benchmarks.model_evaluation import QUANTILES


def _case(name, history, truth):
    return {"name": name,
            "data": {"inputs": [{"item_id": f"{name}/origin-4", "target": history}],
                     "parameters": {"prediction_length": len(truth), "quantile_levels": list(QUANTILES)}},
            "evaluation": {"targets": [truth], "observed": [[True] * len(truth)], "seasonality": 1},
            "provenance": {"fixture": "artificial unit-test values"}}


class _Session:
    """Returns native forecasts, shifted by ``shift`` for the named cases."""

    def __init__(self, shift):
        self.shift = shift

    def post(self, url, json, timeout):
        name = json["data"]["inputs"][0]["item_id"].split("/")[0]
        values = [[value + self.shift.get(name, 0.) for value in row] for row in _EXPECTED[name]]
        body = {"data": {"predictions": [{"item_id": json["data"]["inputs"][0]["item_id"],
                                          **{str(q): values[i] for i, q in enumerate(QUANTILES)}}]}}
        return SimpleNamespace(text="{}", raise_for_status=lambda: None, json=lambda: body)


_EXPECTED = {name: [[float(quantile + step) for step in range(2)] for quantile in range(9)]
             for name in ("first", "second", "third")}


def _run(tmp_path, shift):
    cases = [{"case": _case(name, [1., 2., 3., 4.], [5., 6.]), "expected": [_EXPECTED[name]]}
             for name in ("first", "second", "third")]
    return {"cases": cases, "spec": SimpleNamespace(model_id="unit-test"), "session": _Session(shift),
            "url": "http://unused", "output": tmp_path, "metadata": {"checks": []}}


def test_parity_failure_is_recorded_and_later_cases_still_run(tmp_path):
    run = _run(tmp_path, {"second": 3 * 2 ** -12})  # three FP32 steps near 3,000
    with pytest.raises(pytest.fail.Exception, match="1 of 3 accuracy cases failed:\nsecond: second: series 0"):
        accuracy.test_observed_forecast_quality_preserves_native(run)
    checks = run["metadata"]["checks"]
    assert [check["status"] for check in checks] == ["passed", "failed", "passed"]
    assert "exceeds tolerance" in checks[1]["error"]
    saved = json.loads((tmp_path / "accuracy.json").read_text())["cases"]
    assert [entry["parity"] for entry in saved] == ["passed", "failed", "passed"]
    assert "exceeds tolerance" in saved[1]["parity_error"]
    assert json.loads((tmp_path / "run.json").read_text())["checks"] == checks


def test_matching_cases_pass_without_failure(tmp_path):
    run = _run(tmp_path, {})
    accuracy.test_observed_forecast_quality_preserves_native(run)
    assert [check["status"] for check in run["metadata"]["checks"]] == ["passed"] * 3
    assert all(check["compared_values"] == 18 for check in run["metadata"]["checks"])


def test_interrupts_still_stop_the_run(tmp_path):
    run = _run(tmp_path, {})

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    run["session"].post = interrupted
    with pytest.raises(KeyboardInterrupt):
        accuracy.test_observed_forecast_quality_preserves_native(run)
    assert len(run["metadata"]["checks"]) == 1
