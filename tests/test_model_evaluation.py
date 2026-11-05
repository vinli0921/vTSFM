"""CPU checks for held-out real-data workloads and past-only scoring."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
import hashlib
import io
import shutil
import subprocess
import sys
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from benchmarks.model_evaluation import (
    QUANTILES,
    _causal_fill,
    assetops_cases,
    load_cases,
    score_forecasts,
    seasonal_naive_forecasts,
    validate_case,
)
from benchmarks.prepare_data import COLUMNS, prepare


@pytest.fixture(scope="module")
def prepared_data(tmp_path_factory):
    """Artificial source used only for tests, never evaluation observations."""
    root = tmp_path_factory.mktemp("evaluation-data")
    beginning = datetime(2020, 2, 1)
    rows = []
    for index in range(5116):
        row = {"asset_id": "mp_1", "timestamp": (beginning + timedelta(hours=index)).isoformat()}
        row.update({name: "" if 10 <= index < 570 else index * (channel + 1)
                    for channel, (name, _, _) in enumerate(COLUMNS)})
        rows.append(row)
    source_bytes = json.dumps(rows).encode()
    source_file = root / "unit-test-source.json"
    source_file.write_bytes(source_bytes)
    # Test data uses its own pin, never the real fixture's trusted checksum.
    with patch("benchmarks.prepare_data.SOURCE_SHA256", hashlib.sha256(source_bytes).hexdigest()):
        prepare(root, source_file)
        yield root


def case_for_scores(pasts=None, targets=None, observed=None, seasonality=1):
    pasts = [[0., 1., 2., 3.], [10., 12., 14., 16.]] if pasts is None else pasts
    targets = [[4., 5.], [18., 20.]] if targets is None else targets
    observed = [[True] * len(row) for row in targets] if observed is None else observed
    return {"name": "unit-test-only", "data": {
        "inputs": [{"item_id": f"series-{row}", "target": past} for row, past in enumerate(pasts)],
        "parameters": {"prediction_length": len(targets[0]), "quantile_levels": list(QUANTILES)}},
        "evaluation": {"targets": targets, "observed": observed, "seasonality": seasonality},
        "provenance": {"fixture": "artificial unit-test values, not benchmark data"}}


def test_assetops_origins_are_observed_chronological_and_reproducible(prepared_data):
    cases = assetops_cases(prepared_data, count=5, batch_size=7)
    assert cases == assetops_cases(prepared_data, count=5, batch_size=7)
    assert len({case["provenance"]["origin"] for case in cases}) == 5
    for case in cases:
        origin = case["provenance"]["origin"]
        assert not set(range(origin, origin + 32)) & set(range(10, 570))
        assert all(all(mask) for mask in case["evaluation"]["observed"])
        for channel, (item, truth) in enumerate(zip(case["data"]["inputs"], case["evaluation"]["targets"])):
            last_observed = 9 if 10 <= origin - 1 < 570 else origin - 1
            assert item["target"][-1] == last_observed * (channel + 1)
            assert truth == [position * (channel + 1) for position in range(origin, origin + 32)]
            assert f"/{COLUMNS[channel][0]}/" in item["item_id"]


def test_long_context_uses_only_documented_causal_filled_past(prepared_data):
    case = assetops_cases(prepared_data, context=512, horizon=96, count=1)[0]
    origin = case["provenance"]["origin"]
    assert origin == 570
    assert case["provenance"]["past_imputed_positions"] == list(range(58, 570))
    assert case["data"]["inputs"][0]["target"] == [9.] * 512
    assert case["evaluation"]["targets"][0] == list(range(570, 666))


def test_large_performance_batch_is_explicitly_correlated(prepared_data):
    case = assetops_cases(prepared_data, batch_size=16)[0]
    items = case["data"]["inputs"]
    assert len({item["item_id"] for item in items}) == 16
    assert items[0]["target"] == items[7]["target"]
    assert case["provenance"]["repeated_windows"] == 9
    assert case["provenance"]["performance_only_repetitions"] is True


def test_missing_fixture_never_falls_back_or_downloads(tmp_path):
    with patch("urllib.request.urlopen", side_effect=AssertionError("must not download")):
        with pytest.raises(ValueError, match="Missing prepared AssetOps"):
            assetops_cases(tmp_path)


def test_corrupted_array_is_rejected(prepared_data, tmp_path):
    shutil.copytree(prepared_data, tmp_path, dirs_exist_ok=True)
    with (tmp_path / "data.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assetops_cases(tmp_path)


@pytest.mark.parametrize("field,value,match", [
    ("source", {"commit": "changed"}, "source pin"),
    ("imputation", {"source_row_indices": []}, "imputation provenance"),
])
def test_inconsistent_provenance_is_rejected(prepared_data, tmp_path, field, value, match):
    shutil.copytree(prepared_data, tmp_path, dirs_exist_ok=True)
    manifest = json.loads((tmp_path / "data_manifest.json").read_text())
    manifest[field].update(value)
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        assetops_cases(tmp_path)


def test_changed_source_index_order_is_rejected(prepared_data, tmp_path):
    shutil.copytree(prepared_data, tmp_path, dirs_exist_ok=True)
    indices = list(range(5116))
    indices[100], indices[101] = indices[101], indices[100]
    (tmp_path / "source_row_indices.json").write_text(json.dumps(indices))
    with pytest.raises(ValueError, match="chronological"):
        assetops_cases(tmp_path)


def test_rehashed_changed_observation_still_must_match_original_source(prepared_data, tmp_path):
    shutil.copytree(prepared_data, tmp_path, dirs_exist_ok=True)
    with np.load(tmp_path / "data.npz", allow_pickle=False) as archive:
        series = archive["series"]
    series[0, 1000] += 100
    output = io.BytesIO()
    np.savez_compressed(output, series=series)
    content = output.getvalue()
    (tmp_path / "data.npz").write_bytes(content)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        npy_hash = hashlib.sha256(archive.read("series.npy")).hexdigest()
    manifest = json.loads((tmp_path / "data_manifest.json").read_text())
    manifest["array"].update(sha256=hashlib.sha256(content).hexdigest(), series_npy_sha256=npy_hash)
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="observed values do not match"):
        assetops_cases(tmp_path)


def test_corrupted_original_source_is_rejected(prepared_data, tmp_path):
    shutil.copytree(prepared_data, tmp_path, dirs_exist_ok=True)
    source = tmp_path / "source/metro_pump_1.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="original source checksum"):
        assetops_cases(tmp_path)


def test_no_observed_future_origin_is_an_error(prepared_data):
    with pytest.raises(ValueError, match="distinct origins"):
        assetops_cases(prepared_data, context=5100, horizon=32)


def test_metrics_use_median_and_nine_quantiles_with_past_only_scale():
    case = case_for_scores()
    predictions = np.repeat(np.array([[3., 3.], [16., 16.]])[:, None, :], 9, axis=1)
    scores = score_forecasts(case, predictions)
    assert scores["MAE"] == 2.25
    assert scores["MASE"] == 1.5
    # Identical point forecasts in every symmetric quantile make SQL equal MASE.
    assert scores["SQL"] == pytest.approx(1.5)
    assert scores["counts"]["valid_scaled_series"] == 2
    assert [row["seasonal_scale"] for row in scores["per_series"]] == [1., 2.]
    changed = deepcopy(case)
    changed["evaluation"]["targets"][0] = [400., 500.]
    assert [row["seasonal_scale"] for row in score_forecasts(changed, predictions)["per_series"]] == [1., 2.]


def test_each_quantile_contributes_and_factor_two_is_applied():
    case = case_for_scores(pasts=[[0., 1.]], targets=[[2.]])
    predictions = np.zeros((1, 9, 1))
    predictions[:, 4, :] = 2.
    scores = score_forecasts(case, predictions)
    assert scores["MAE"] == 0.
    assert scores["MASE"] == 0.
    assert scores["SQL"] == pytest.approx(16 / 9)


def test_missing_future_labels_are_masked_in_all_metrics():
    case = case_for_scores(pasts=[[0., 1., 2.]], targets=[[3., None]], observed=[[True, False]])
    predictions = np.repeat(np.array([[[2., 1e10]]]), 9, axis=1)
    scores = score_forecasts(case, predictions)
    assert scores["MAE"] == scores["MASE"] == scores["SQL"] == pytest.approx(1.)
    assert scores["counts"]["observed_targets"] == 1


def test_zero_scale_and_unobserved_series_are_explicit_json_null():
    case = case_for_scores(pasts=[[2., 2., 2.], [0., 1., 2.]], targets=[[5., 6.], [None, None]],
                           observed=[[True, True], [False, False]])
    scores = score_forecasts(case, np.zeros((2, 9, 2)))
    assert scores["MAE"] == 5.5
    assert scores["MASE"] is None and scores["SQL"] is None
    assert scores["counts"] == {"series": 2, "scored_series": 1, "observed_targets": 2,
                                 "valid_scaled_series": 0, "zero_scale_series": 1,
                                 "insufficient_history_series": 0, "unobserved_series": 1}
    json.dumps(scores, allow_nan=False)


def test_short_past_has_no_silent_seasonality_fallback():
    case = case_for_scores(pasts=[[0., 1.]], targets=[[2.]], seasonality=2)
    scores = score_forecasts(case, np.zeros((1, 9, 1)))
    assert scores["MAE"] == 2.
    assert scores["MASE"] is None
    assert scores["counts"]["insufficient_history_series"] == 1


def test_seasonal_naive_repeats_last_season_and_alignment_matters():
    case = case_for_scores(pasts=[[1., 2., 3., 4.], [10., 20., 30., 40.]],
                           targets=[[3., 4., 3.], [30., 40., 30.]], seasonality=2)
    forecasts = seasonal_naive_forecasts(case)
    assert forecasts.shape == (2, 9, 3)
    assert score_forecasts(case, forecasts)["MAE"] == 0.
    assert score_forecasts(case, forecasts[::-1])["MAE"] > 0.
    permuted = deepcopy(case)
    permuted["data"]["inputs"].reverse()
    permuted["evaluation"]["targets"].reverse()
    permuted["evaluation"]["observed"].reverse()
    assert score_forecasts(permuted, forecasts[::-1])["MAE"] == 0.


@pytest.mark.parametrize("bad", [np.zeros((2, 2, 9)), np.full((2, 9, 2), np.nan)])
def test_invalid_forecasts_fail_instead_of_silently_scoring(bad):
    with pytest.raises(ValueError, match="finite with shape"):
        score_forecasts(case_for_scores(), bad)


def test_manifest_round_trip_and_observed_target_validation(tmp_path):
    case = case_for_scores()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [case], "provenance": {"fixture": "unit test"}}))
    assert load_cases(path) == [case]
    broken = deepcopy(case)
    broken["evaluation"]["targets"][0][0] = None
    with pytest.raises(ValueError, match="Observed targets"):
        validate_case(broken)


def test_causal_fill_never_uses_a_future_observation():
    assert _causal_fill([1., None, float("nan"), 9.]) == ([1., 1., 1., 9.], [1, 2])
    with pytest.raises(ValueError, match="starts with a missing"):
        _causal_fill([None, 9.])
    assert _causal_fill(np.array([1., np.nan, 9.], dtype="float32")) == ([1., 1., 9.], [1])


def test_prepare_cli_writes_new_manifest_and_preserves_existing_output(prepared_data, tmp_path):
    # The fake fixture's checksum is intentionally local to the test process.
    from benchmarks.model_evaluation import main

    output = tmp_path / "evaluation.json"
    arguments = ["model_evaluation", "prepare", "--data-dir", str(prepared_data), "--output", str(output)]
    with patch.object(sys, "argv", arguments):
        main()
    cases = load_cases(output)
    assert len(cases) == 5
    assert all(len(case["data"]["inputs"]) == 7 for case in cases)
    original = output.read_bytes()
    with patch.object(sys, "argv", arguments), pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_help_and_import_are_cpu_safe():
    code = "import sys; import benchmarks.model_evaluation; assert not {'torch', 'vllm', 'fev', 'numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    result = subprocess.run([sys.executable, "-m", "benchmarks.model_evaluation", "prepare", "--help"], check=True, capture_output=True, text=True)
    assert "--fev" in result.stdout
