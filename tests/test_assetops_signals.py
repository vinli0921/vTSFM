"""CPU checks for pinned AssetOps source fetching and signal accuracy cases.

All series here are artificial unit-test values with their own local pins,
never benchmark observations.
"""

from datetime import datetime, timedelta
import hashlib
import json
import sys
from unittest.mock import patch

import pytest

from benchmarks import prepare_data
from benchmarks.model_evaluation import load_cases, main, signal_cases


START = datetime(2020, 6, 1)


def _time_rows():
    """32 nominal 15-minute slots with jitter, gaps and a late duplicate-time row."""
    rows = []
    for slot in range(32):
        jitter = (-60, 0, 45)[slot % 3]
        row = {"asset_id": "a1", "timestamp": (START + timedelta(seconds=900 * slot + jitter)).isoformat(),
               "flow": float(slot) if slot % 4 else 0.0}
        if slot == 5:
            del row["flow"]          # absent column: slot 5 stays missing
        if slot == 9:
            row["flow"] = ""         # empty value: filled by the late row below
        rows.append(row)
    # A late event row (out of order in the file) that is the only reading for slot 9.
    rows.append({"asset_id": "a1", "timestamp": (START + timedelta(seconds=900 * 9 + 30)).isoformat(), "flow": 9.0})
    return rows


def _cycle_rows():
    return [{"asset_id": "h1", "cycle": cycle, "pressure": 100. + cycle} for cycle in range(1, 41)]


@pytest.fixture
def sources(tmp_path):
    """A local pin set plus a fake AssetOpsBench checkout holding its files."""
    payloads = {"time.json": json.dumps(_time_rows()).encode(), "cycle.json": json.dumps(_cycle_rows()).encode()}
    config = {
        "repository": "example/unit-test", "commit": "0" * 40, "directory": "iot",
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()},
        "signals": {
            "t/flow": {"file": "time.json", "column": "flow", "asset_id": "a1",
                       "axis": {"field": "timestamp", "step_seconds": 900}, "seasonality": 4},
            "c/pressure": {"file": "cycle.json", "column": "pressure", "asset_id": "h1",
                           "axis": {"field": "cycle", "step": 1}, "seasonality": 1},
        },
    }
    checkout = tmp_path / "checkout"
    (checkout / "iot").mkdir(parents=True)
    for name, data in payloads.items():
        (checkout / "iot" / name).write_bytes(data)
    with patch.object(prepare_data, "SOURCES", config):
        yield {"config": config, "checkout": checkout, "data": tmp_path / "data"}


def _no_download(*args, **kwargs):
    raise AssertionError("must not download")


def test_real_pin_list_is_valid_and_keeps_the_metro_fixture_pin():
    sources = prepare_data.load_sources()
    assert prepare_data.SOURCE_SHA256 == sources["files"]["metro_pump_1.json"]["sha256"]
    assert prepare_data.SOURCE_URL.endswith(f"/{sources['commit']}/{sources['directory']}/metro_pump_1.json")
    assert {signal["file"] for signal in sources["signals"].values()} <= set(sources["files"])


def test_fetch_copies_once_then_reuses_verified_files(sources):
    first = prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    assert first == {"time.json": "copied", "cycle.json": "copied"}
    with patch("urllib.request.urlopen", side_effect=_no_download):
        assert prepare_data.fetch_sources(sources["data"]) == {"time.json": "present", "cycle.json": "present"}


def test_fetch_downloads_missing_files_from_the_pinned_commit(sources):
    data = (sources["checkout"] / "iot/cycle.json").read_bytes()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return data

    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        assert prepare_data.fetch_sources(sources["data"], ["cycle.json"]) == {"cycle.json": "downloaded"}
    assert urlopen.call_args.args[0].full_url == \
        f"https://raw.githubusercontent.com/example/unit-test/{'0' * 40}/iot/cycle.json"


def test_fetch_rejects_changed_cache_unpinned_names_and_changed_checkout(sources):
    prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    cached = sources["data"] / "source/time.json"
    cached.write_bytes(cached.read_bytes() + b" ")
    with pytest.raises(ValueError, match="pinned SHA256"):
        prepare_data.fetch_sources(sources["data"])
    with pytest.raises(ValueError, match="Unpinned"):
        prepare_data.fetch_sources(sources["data"], ["other.json"])
    (sources["checkout"] / "iot/cycle.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_data.fetch_sources(sources["data"] / "fresh", ["cycle.json"], checkout=sources["checkout"])


def test_timestamp_signal_is_gridded_with_missing_slots_masked(sources):
    prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    signal = prepare_data.load_signal(sources["data"], "t/flow")
    assert len(signal["values"]) == 32
    assert signal["observed"][5] is False and signal["values"][5] is None
    assert signal["values"][9] == 9.0  # the late out-of-order reading lands in its own slot
    assert signal["values"][8] == 0.0 and signal["values"][31] == 31.0
    assert signal["axis"][1] == (START + timedelta(seconds=900)).isoformat()
    provenance = signal["provenance"]
    assert provenance["missing_slot_indices"] == [5]
    assert provenance["rows_without_value"] == 2 and provenance["max_grid_offset_seconds"] == 60


def test_counter_signal_uses_one_slot_per_cycle(sources):
    prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    signal = prepare_data.load_signal(sources["data"], "c/pressure")
    assert signal["values"] == [100. + cycle for cycle in range(1, 41)]
    assert signal["axis"][0] == "1" and all(signal["observed"])


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.append(dict(rows[3])), "share slot"),
    (lambda rows: rows[2].update(timestamp=(START + timedelta(seconds=900 * 2 + 300)).isoformat()), "from the 900s grid"),
    (lambda rows: rows[2].pop("timestamp"), r"Invalid timestamp at time\.json row 2"),
    (lambda rows: rows[4].update(flow=True), "Non-numeric"),
    (lambda rows: rows[4].update(asset_id="other"), "Unexpected asset"),
])
def test_irregular_or_invalid_source_rows_are_rejected(sources, change, match):
    rows = _time_rows()
    change(rows)
    data = json.dumps(rows).encode()
    sources["config"]["files"]["time.json"]["sha256"] = hashlib.sha256(data).hexdigest()
    (sources["data"] / "source").mkdir(parents=True)
    (sources["data"] / "source/time.json").write_bytes(data)
    with pytest.raises(ValueError, match=match):
        prepare_data.load_signal(sources["data"], "t/flow")


def test_signal_cases_score_only_recorded_futures_with_causal_history(sources):
    prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    with patch("urllib.request.urlopen", side_effect=_no_download):
        case, = signal_cases(sources["data"], ["t/flow"], context=8, horizon=4, windows=3)
    origins = case["provenance"]["origins"]
    assert origins == sorted(set(origins)) and len(origins) == 3
    assert case["provenance"]["past_imputed_counts"] == [int(origin - 8 <= 5 < origin) for origin in origins]
    assert case["evaluation"]["seasonality"] == 4
    for origin, item, truth in zip(origins, case["data"]["inputs"], case["evaluation"]["targets"]):
        assert item["item_id"] == f"t/flow/origin-{origin}"
        assert 5 not in range(origin, origin + 4)  # a missing slot is never a scored label
        assert None not in truth and len(item["target"]) == 8
        if 5 in range(origin - 8, origin):
            assert item["target"][5 - (origin - 8)] == 0.0  # slot 4's reading carried forward
    assert case["provenance"]["truth_policy"].startswith("Only windows")


def test_signal_cases_need_prepared_sources_and_enough_origins(sources):
    with patch("urllib.request.urlopen", side_effect=_no_download):
        with pytest.raises(ValueError, match=r"python -m benchmarks\.prepare_data --assetops-signals"):
            signal_cases(sources["data"], ["c/pressure"])
    prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    with pytest.raises(ValueError, match="origins"):
        signal_cases(sources["data"], ["c/pressure"], context=30, horizon=8, windows=5)


def test_prepare_cli_writes_signal_cases_instead_of_the_default_fixture(sources, tmp_path):
    prepare_data.fetch_sources(sources["data"], checkout=sources["checkout"])
    output = tmp_path / "signals.json"
    arguments = ["model_evaluation", "prepare", "--data-dir", str(sources["data"]), "--output", str(output),
                 "--assetops-signals", "--signal-context", "8", "--signal-horizon", "4", "--signal-windows", "2"]
    with patch.object(sys, "argv", arguments):
        main()
    cases = load_cases(output)
    assert [case["name"] for case in cases] == ["signal-t-flow-c8-h4", "signal-c-pressure-c8-h4"]
    manifest = json.loads(output.read_text())["provenance"]
    assert manifest["assetops_signals"] == ["t/flow", "c/pressure"] and manifest["signal_windows"] == 2
