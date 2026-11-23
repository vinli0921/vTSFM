"""CPU checks for the accuracy-run summary table; scores are artificial."""

import json

import pytest

from benchmarks.accuracy_table import (collect, collect_windows, compare, main, markdown, model_spread,
                                       precision_deltas)


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


def test_parity_failures_are_flagged_in_the_table(tmp_path):
    _write_run(tmp_path / "run", "toto2", "float32", [_series("s/x/origin-1", 1.0)])
    path = tmp_path / "run/toto2/accuracy.json"
    report = json.loads(path.read_text())
    report["cases"][0]["parity"] = "failed"
    path.write_text(json.dumps(report))
    row, = collect([tmp_path / "run"])
    assert row["parity_failed_cases"] == 1
    assert "(parity failed)" in markdown([row])


def test_observed_scale_rescales_exactly_and_reports_filled_history(tmp_path):
    from benchmarks.model_evaluation import QUANTILES, score_forecasts

    case = {"name": "signal-s-x", "data": {"inputs": [{"item_id": "s/x/origin-108",
                                                         "target": [0., 1., 3., 3., 3., 1., 7., 4.]}],
                                           "parameters": {"prediction_length": 2, "quantile_levels": list(QUANTILES)}},
            "evaluation": {"targets": [[5., 6.]], "observed": [[True, True]], "seasonality": 2},
            "provenance": {"origins": [108], "past_imputed_positions": [[103]]}}
    forecast = [[[4. + q / 10, 4.5] for q in range(9)]]
    every = score_forecasts(case, forecast)["per_series"]
    _write_run(tmp_path / "run", "toto2", "float32", every, case="signal-s-x")
    (tmp_path / "run/toto2/cases.json").write_text(json.dumps({"cases": [case]}))

    default, = collect([tmp_path / "run"])
    assert default["filled_history_pct"] == pytest.approx(12.5) and default["scale"] == "all"
    observed, = collect([tmp_path / "run"], "observed")
    direct = score_forecasts(case, forecast, scale="observed")
    assert observed["MASE"] == pytest.approx(direct["MASE"]) and observed["SQL"] == pytest.approx(direct["SQL"])
    assert observed["MAE"] == default["MAE"]
    assert "recorded history pairs only" in markdown([observed]) and "| 12.5% |" in markdown([observed])

    assert main([str(tmp_path / "run"), "--scale", "observed", "--output", str(tmp_path / "out")]) == 0
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "accuracy_table_observed_scale.csv", "accuracy_table_observed_scale.md",
        "accuracy_table_observed_scale_comparison.csv", "accuracy_table_observed_scale_windows.csv"]


def test_observed_scale_requires_the_run_cases(tmp_path):
    _write_run(tmp_path / "run", "toto2", "float32", [_series("s/x/origin-1", 1.0)])
    with pytest.raises(ValueError, match="needs .*cases.json"):
        collect([tmp_path / "run"], "observed")
    row, = collect([tmp_path / "run"])
    assert row["filled_history_pct"] is None and "| n/a |" in markdown([row])


def _window(item_id, mae, scale, sql=None):
    """One per-series score whose MASE/SQL follow from MAE and the seasonal scale."""
    status = "valid" if scale else "zero"
    return {"item_id": item_id, "MAE": mae, "MASE": mae / scale if scale else None,
            "SQL": (mae if sql is None else sql) / scale if scale else None,
            "seasonal_scale": scale, "scale_status": status, "observed_targets": 2}


def _write_scored_run(root, adapter, native, naive, history, case="signal-s-x"):
    _write_run(root, adapter, "float32", native, naive=naive, case=case)
    path = root / adapter / "accuracy.json"
    report = json.loads(path.read_text())
    report["cases"][0]["mean_history"] = {"per_series": history}
    path.write_text(json.dumps(report))


def test_robust_aggregates_resist_a_tiny_scale_window_and_count_undefined_scales(tmp_path):
    native = [_window("s/x/origin-10", 1.0, 1.0), _window("s/x/origin-20", 1.0, 1.0),
              _window("s/x/origin-30", 0.1, 0.001), _window("s/x/origin-40", 2.0, 0.0)]
    _write_run(tmp_path / "run", "toto2", "float32", native)
    row, = collect([tmp_path / "run"])
    assert row["series"] == 4 and row["scaled_series"] == 3 and row["zero_scale_series"] == 1
    assert row["MASE"] == pytest.approx(34.0)  # The tiny-scale window dominates the mean of ratios.
    assert row["MASE_ratio_of_means"] == pytest.approx(2.1 / 2.001)
    assert row["MASE_geometric_mean"] == pytest.approx(100 ** (1 / 3)) and row["geometric_mean_windows"] == 3
    assert row["SQL_ratio_of_means"] == pytest.approx(2.1 / 2.001)
    assert row["mean_history_MASE"] is None and row["off_state_windows"] == 0


def _comparison_run(root, ratios, windows=6, signal="s/x"):
    """Models whose MAE is ``ratio`` x a shared per-window error; both baselines are worse."""
    base = [1.0 + index for index in range(windows)]
    scales = [0.5 + index % 3 for index in range(windows)]
    items = [f"{signal}/origin-{10 * index}" for index in range(windows)]
    naive = [_window(item, 3 * error, scale) for item, error, scale in zip(items, base, scales)]
    history = [_window(item, 4 * error, scale) for item, error, scale in zip(items, base, scales)]
    for adapter, ratio in ratios.items():
        native = [_window(item, ratio * error, scale) for item, error, scale in zip(items, base, scales)]
        _write_scored_run(root, adapter, native, naive, history)


def test_paired_bootstrap_is_reproducible_and_scores_every_competitor_on_the_same_draw(tmp_path):
    _comparison_run(tmp_path / "run", {"timesfm25": 1.0, "toto2": 2.0})
    windows = collect_windows([tmp_path / "run"])
    rows = compare(windows, resamples=300, seed=7)
    assert rows == compare(windows, resamples=300, seed=7)
    by_name = {row["competitor"]: row for row in rows}
    assert set(by_name) == {"timesfm25", "toto2", "seasonal_naive", "mean_history"}
    lowest = by_name["timesfm25"]
    assert lowest["claim"] == "lowest; every other competitor higher" and lowest["probability_lowest"] == 1.0
    # Paired draws keep each competitor's ratio to the lowest exact; unpaired draws would not.
    for name, ratio in [("toto2", 2.0), ("seasonal_naive", 3.0), ("mean_history", 4.0)]:
        row = by_name[name]
        assert row["ratio_to_lowest"] == pytest.approx(ratio)
        assert row["ratio_to_lowest_low"] == pytest.approx(ratio) == row["ratio_to_lowest_high"]
        assert row["claim"] == "higher than lowest"
    for row in rows:
        assert row["MASE_ratio_of_means_low"] <= row["MASE_ratio_of_means"] <= row["MASE_ratio_of_means_high"]
        assert row["MASE_geometric_mean_low"] <= row["MASE_geometric_mean"] <= row["MASE_geometric_mean_high"]
    assert [by_name[name]["mean_rank"] for name in ("timesfm25", "toto2", "seasonal_naive", "mean_history")] == [
        1.0, 2.0, 3.0, 4.0]


def test_claims_need_enough_windows_distinct_intervals_and_a_forecastable_signal(tmp_path):
    _comparison_run(tmp_path / "few", {"timesfm25": 1.0, "toto2": 2.0}, windows=4)
    assert {row["claim"] for row in compare(collect_windows([tmp_path / "few"]), resamples=50)} == {
        "no claim: fewer than 5 windows"}
    _comparison_run(tmp_path / "close", {"timesfm25": 1.0, "toto2": 1.0})
    close = {row["competitor"]: row["claim"] for row in compare(collect_windows([tmp_path / "close"]), resamples=50)}
    assert close["timesfm25"] == "lowest; not distinguishable from toto2"
    assert close["toto2"] == "not distinguishable from lowest"
    _comparison_run(tmp_path / "noise", {"timesfm25": 1.0, "toto2": 2.0}, signal="chiller_6/supply_temperature")
    windows = collect_windows([tmp_path / "noise"])
    assert {row["claim"] for row in compare(windows, resamples=50)} == {"no claim: unforecastable signal"}
    table = collect([tmp_path / "noise"], windows=windows)
    assert all(row["unforecastable"] for row in table)
    assert "(unforecastable)" in markdown(table) and "excluded from ranking claims" in markdown(table)


def _source_case(name, items, histories, targets):
    from benchmarks.model_evaluation import QUANTILES

    return {"name": name, "data": {"inputs": [{"item_id": item, "target": history}
                                              for item, history in zip(items, histories)],
                                   "parameters": {"prediction_length": len(targets[0]),
                                                  "quantile_levels": list(QUANTILES)}},
            "evaluation": {"targets": targets, "observed": [[True] * len(row) for row in targets], "seasonality": 1},
            "provenance": {"origins": [int(item.rsplit("-", 1)[1]) for item in items],
                           "past_imputed_positions": [[] for _ in items]}}


def test_off_state_flags_recomputed_mean_history_and_window_csv(tmp_path):
    items = [f"s/x/origin-{index}" for index in (4, 8, 12, 16, 20, 24)]
    histories = [[1., 3., 1., 3.]] * 6  # Mean 2, one-step scale 2.
    targets = [[0., 0.], [0., 5.], [2., 2.], [4., 2.], [1., 3.], [2., 6.]]
    native = [_window(item, 1.0, 2.0) for item in items]
    _write_run(tmp_path / "run", "toto2", "float32", native, naive=[_window(item, 3.0, 2.0) for item in items])
    (tmp_path / "run/toto2/cases.json").write_text(json.dumps(
        {"cases": [_source_case("signal-x", items, histories, targets)]}))
    windows = collect_windows([tmp_path / "run"])  # No recorded mean_history: recomputed from cases.json.
    assert [window["off_state"] for window in windows] == [True, True, False, False, False, False]
    assert [window["mean_history"]["MAE"] for window in windows] == pytest.approx([2., 2.5, 0., 1., 1., 2.])
    row, = collect([tmp_path / "run"], windows=windows)
    assert row["off_state_windows"] == 2 and row["mean_history_MASE"] == pytest.approx(8.5 / 6 / 2)
    assert "2 off-state" in markdown([row])
    subsets = {(entry["subset"], entry["competitor"]): entry for entry in compare(windows, resamples=20)}
    assert subsets[("all", "toto2")]["windows"] == 6
    assert subsets[("excluding_off_state", "toto2")]["windows"] == 4

    assert main([str(tmp_path / "run"), "--resamples", "20", "--output", str(tmp_path / "out")]) == 0
    lines = (tmp_path / "out/accuracy_table_windows.csv").read_text().splitlines()
    assert lines[0].startswith("model,dtype,signal,case,item_id,origin,") and len(lines) == 7
    assert "mean_history_MASE" in lines[0] and ",True,False," in lines[1]
    comparison = (tmp_path / "out/accuracy_table_comparison.csv").read_text().splitlines()
    assert comparison[0].startswith("dtype,signal,subset,competitor,") and len(comparison) == 1 + 2 * 3
