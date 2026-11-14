"""CPU checks for the accuracy-run summary table; scores are artificial."""

import json

import pytest

from benchmarks.accuracy_table import collect, main, markdown, model_spread, precision_deltas


def _series(item_id, mase, sql=None, status="valid"):
    return {"item_id": item_id, "MAE": mase, "MASE": mase if status == "valid" else None,
            "SQL": (mase if sql is None else sql) if status == "valid" else None, "scale_status": status}


def _write_run(root, adapter, precision, native, served=None, naive=None, case="signal-x"):
    folder = root / adapter
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"precision": precision}))
    served = native if served is None else served
    naive = native if naive is None else naive
    scores = lambda rows: {"per_series": rows}  # noqa: E731
    (folder / "accuracy.json").write_text(json.dumps({"cases": [
        {"case": case, "native": scores(native), "vtsfm": scores(served), "seasonal_naive": scores(naive)}]}))


def test_groups_series_by_signal_and_averages_only_defined_scores(tmp_path):
    native = [_series("a/flow/origin-10", 1.0), _series("a/flow/origin-20", 3.0),
              _series("a/flow/origin-30", 9.0, status="zero"), _series("b/temp/origin-10", 0.5)]
    served = [dict(row, SQL=None if row["SQL"] is None else row["SQL"] + 1e-6) for row in native]
    naive = [_series(row["item_id"], 2.0) for row in native]
    _write_run(tmp_path / "fp32", "toto2", "float32", native, served, naive)
    flow, temp = collect([tmp_path / "fp32"])
    assert (flow["signal"], flow["dtype"], flow["model"]) == ("a/flow", "fp32", "toto2")
    assert flow["series"] == 3 and flow["scaled_series"] == 2 and flow["zero_scale_series"] == 1
    assert flow["MASE"] == 2.0 and flow["SQL"] == 2.0 and flow["MAE"] == pytest.approx(13 / 3)
    assert flow["naive_MASE"] == 2.0
    assert flow["max_abs_vtsfm_minus_native_SQL"] == pytest.approx(1e-6)
    assert temp["signal"] == "b/temp" and temp["series"] == 1


def test_combines_runs_into_dtype_deltas_and_model_spread(tmp_path):
    rows = lambda value: [_series("s/x/origin-1", value)]  # noqa: E731
    _write_run(tmp_path / "fp32", "toto2", "float32", rows(1.0))
    _write_run(tmp_path / "fp32", "moirai2", "float32", rows(2.0))
    _write_run(tmp_path / "bf16", "toto2", "bf16", rows(1.1))
    table = collect([tmp_path / "fp32", tmp_path / "bf16"])
    assert [(row["model"], row["dtype"]) for row in table] == [("toto2", "fp32"), ("toto2", "bf16"), ("moirai2", "fp32")]
    delta, = precision_deltas(table)
    assert delta["dtype"] == "bf16" and delta["delta_MASE"] == pytest.approx(0.1)
    assert delta["relative_SQL"] == pytest.approx(0.1)
    spread, = model_spread(table)
    assert spread["best_MASE"] == "toto2 1.000" and spread["worst_over_best_MASE"] == 2.0
    text = markdown(table)
    assert "| s/x | toto2 | bf16 |" in text and "+10.00%" in text


def test_fev_items_group_by_case_without_window_suffix(tmp_path):
    _write_run(tmp_path / "run", "timesfm25", "fp32", [_series("T1", 1.0)], case="fev-smoke-m4_hourly-window-1")
    row, = collect([tmp_path / "run"])
    assert row["signal"] == "fev-smoke-m4_hourly"


def test_cli_writes_markdown_and_csv_and_rejects_empty_runs(tmp_path, capsys):
    _write_run(tmp_path / "run", "toto2", "float32", [_series("s/x/origin-1", 1.0)])
    assert main([str(tmp_path / "run"), "--output", str(tmp_path / "run")]) == 0
    assert "| s/x | toto2 | fp32 |" in capsys.readouterr().out
    assert (tmp_path / "run/accuracy_table.md").is_file()
    assert (tmp_path / "run/accuracy_table.csv").read_text().startswith("model,dtype,signal,")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No <adapter>/accuracy.json"):
        collect([tmp_path / "empty"])
