"""Real-data numerical workloads and small, explicit forecast-quality checks.

Cases keep model inputs separate from held-out targets. Imports and ``--help``
need only the standard library; NumPy and optional fev are loaded when used.
This module does not download AssetOps data or load a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import zipfile

from benchmarks import prepare_data


QUANTILES = [index / 10 for index in range(1, 10)]
FEV_VERSION = "0.10.0"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_case(case):
    """Validate the portable univariate request/ground-truth boundary."""
    if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"]:
        raise ValueError("Each case needs a nonempty name")
    data, evaluation = case.get("data"), case.get("evaluation")
    if not isinstance(data, dict) or not isinstance(evaluation, dict):
        raise ValueError("Each case needs data and evaluation objects")
    if set(data) != {"inputs", "parameters"}:
        raise ValueError("Case data supports only inputs and parameters")
    parameters, inputs = data.get("parameters"), data.get("inputs")
    if not isinstance(parameters, dict) or not isinstance(inputs, list) or not inputs:
        raise ValueError("Each case needs parameters and nonempty inputs")
    horizon = _positive_integer(parameters.get("prediction_length"), "prediction_length")
    if horizon > 1024 or parameters.get("quantile_levels") != QUANTILES:
        raise ValueError("Cases require horizon <= 1024 and all nine increasing quantile tenths")
    if set(parameters) != {"prediction_length", "quantile_levels"}:
        raise ValueError("Only prediction_length and quantile_levels are supported")
    ids = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"target", "item_id"}:
            raise ValueError("Each independent input needs only target and item_id")
        if not isinstance(item["item_id"], str) or not item["item_id"]:
            raise ValueError("item_id must be a nonempty string")
        ids.append(item["item_id"])
        if not isinstance(item["target"], list) or not item["target"] or not all(_finite(x) for x in item["target"]):
            raise ValueError("Input histories must be finite univariate arrays")
    if len(set(ids)) != len(ids):
        raise ValueError("Case item IDs must be unique")
    targets, observed = evaluation.get("targets"), evaluation.get("observed")
    if not isinstance(targets, list) or not isinstance(observed, list) or len(targets) != len(inputs) or len(observed) != len(inputs):
        raise ValueError("Targets and observed masks must align with input rows")
    for target, mask in zip(targets, observed):
        if not isinstance(target, list) or not isinstance(mask, list) or len(target) != horizon or len(mask) != horizon:
            raise ValueError("Targets and masks must have prediction_length values")
        if any(not isinstance(flag, bool) for flag in mask):
            raise ValueError("Observed masks must contain booleans")
        if any((flag or value is not None) and not _finite(value) for value, flag in zip(target, mask)):
            raise ValueError("Observed targets must be finite; missing targets must be null or finite")
    _positive_integer(evaluation.get("seasonality"), "seasonality")
    if not isinstance(case.get("provenance"), dict) or not case["provenance"]:
        raise ValueError("Each case needs provenance")
    return case


def load_cases(path):
    """Load a prepared JSON manifest; never download data as a fallback."""
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("Evaluation manifest needs a nonempty cases list")
    if not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]:
        raise ValueError("Evaluation manifest needs provenance")
    cases = [validate_case(case) for case in manifest["cases"]]
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case names must be unique")
    return cases


def _assetops_data(data_dir):
    import numpy as np

    root = Path(data_dir)
    required = ["data.npz", "data_manifest.json", "source_row_indices.json", "timestamps.json", "source/metro_pump_1.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing prepared AssetOps data: {', '.join(missing)}; run benchmarks.prepare_data first")
    manifest_bytes = (root / "data_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {"commit": prepare_data.COMMIT, "path": prepare_data.SOURCE_PATH,
                "url": prepare_data.SOURCE_URL, "sha256": prepare_data.SOURCE_SHA256,
                "repository": "IBM/AssetOpsBench"}
    if any(manifest.get("source", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("AssetOps source pin does not match benchmarks.prepare_data")
    source_content = (root / "source/metro_pump_1.json").read_bytes()
    if hashlib.sha256(source_content).hexdigest() != prepare_data.SOURCE_SHA256:
        raise ValueError("AssetOps original source checksum mismatch")
    source_rows = json.loads(source_content)
    if not isinstance(source_rows, list) or len(source_rows) != 5116:
        raise ValueError("AssetOps original source must contain the pinned 5,116 rows")
    metadata = manifest.get("array", {})
    content = (root / "data.npz").read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata.get("sha256"):
        raise ValueError("AssetOps array checksum mismatch")
    with zipfile.ZipFile(root / "data.npz") as archive:
        if archive.namelist() != ["series.npy"] or hashlib.sha256(archive.read("series.npy")).hexdigest() != metadata.get("series_npy_sha256"):
            raise ValueError("AssetOps series checksum mismatch")
    with np.load(root / "data.npz", allow_pickle=False) as archive:
        series = archive["series"]
    if (metadata.get("file"), metadata.get("key"), metadata.get("shape"), metadata.get("dtype")) != ("data.npz", "series", [7, 5116], "float32"):
        raise ValueError("Unexpected pinned AssetOps array metadata")
    if series.shape != (7, 5116) or series.dtype != np.dtype("float32") or not np.isfinite(series).all():
        raise ValueError("AssetOps data must contain seven finite float32 channels with 5,116 positions")
    columns = manifest.get("columns", [])
    if [(column.get("array_row"), column.get("name")) for column in columns] != list(enumerate(name for name, _, _ in prepare_data.COLUMNS)):
        raise ValueError("AssetOps channel metadata does not match the pinned channels")
    index_bytes = (root / "source_row_indices.json").read_bytes()
    indices = json.loads(index_bytes)
    if indices != list(range(5116)) or any(type(index) is not int for index in indices):
        raise ValueError("AssetOps source row indices must preserve the pinned chronological hourly grid")
    imputation = manifest.get("imputation", {})
    imputed = imputation.get("source_row_indices")
    if (not isinstance(imputed, list) or len(imputed) != 560
            or any(type(index) is not int or not 0 < index < 5116 for index in imputed)
            or sorted(set(imputed)) != imputed or imputation.get("imputed_rows") != 560
            or imputation.get("observed_source_values_preserved_before_float32_cast") is not True
            or imputation.get("method") != "causal forward-fill from last observed sensor values; no backward-fill"
            or manifest.get("synthetically_generated_observations") != 0):
        raise ValueError("AssetOps causal imputation provenance is missing or inconsistent")
    for index in imputed:
        if not np.array_equal(series[:, index], series[:, index - 1]):
            raise ValueError("AssetOps imputed history is not causal forward-fill")
    timestamp_bytes = (root / "timestamps.json").read_bytes()
    timestamps = json.loads(timestamp_bytes)
    if not isinstance(timestamps, list) or len(timestamps) != 5116:
        raise ValueError("AssetOps timestamps must align with source rows")
    times = [datetime.fromisoformat(value) for value in timestamps]
    if any((end - start).total_seconds() != 3600 for start, end in zip(times, times[1:])):
        raise ValueError("AssetOps timestamps are not a chronological hourly grid")
    if imputation.get("timestamps") != [timestamps[index] for index in imputed]:
        raise ValueError("AssetOps imputation timestamps and source indices disagree")
    missing_source_rows = []
    for index, row in enumerate(source_rows):
        values = [row.get(name) for name, _, _ in prepare_data.COLUMNS]
        if row.get("asset_id") != "mp_1" or row.get("timestamp") != timestamps[index]:
            raise ValueError("AssetOps array time axis does not match the original source")
        if all(value is None or value == "" for value in values):
            missing_source_rows.append(index)
        elif not all(_finite(value) for value in values) or not np.array_equal(series[:, index], np.asarray(values, dtype="float32")):
            raise ValueError("AssetOps observed values do not match the pinned source")
    if missing_source_rows != imputed:
        raise ValueError("AssetOps imputation indices do not match the pinned source's missing labels")
    observed = np.ones(series.shape[1], dtype=bool)
    observed[imputed] = False
    provenance = {
        "dataset": "AssetOpsBench MetroPT compressor fixture", "source": manifest["source"],
        "array_sha256": metadata["sha256"], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_row_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "timestamps_sha256": hashlib.sha256(timestamp_bytes).hexdigest(),
        "history_policy": "Finite observed values with documented causal forward-fill in past only",
        "truth_policy": "Only originally observed, non-imputed future labels are scored",
        "sampling_unit": "Correlated channels/windows from one compressor; not independent assets",
    }
    return series, observed, timestamps, provenance


def assetops_cases(data_dir, *, count=2, context=128, horizon=32, batch_size=2):
    """Build deterministic observed-label windows from the prepared pinned data.

    Inputs are independent univariate requests with shape [N, context]; truth
    has shape [N, horizon]. Histories may contain documented causal fills.
    Batches above seven cycle channels at the same origin for timing workloads,
    and explicitly record their duplicate, correlated-window semantics.
    """
    for value, name in [(count, "count"), (context, "context"), (horizon, "horizon"), (batch_size, "batch_size")]:
        _positive_integer(value, name)
    if horizon > 1024:
        raise ValueError("prediction_length must be <= 1024")
    series, observed, timestamps, provenance = _assetops_data(data_dir)
    origins = [origin for origin in range(context, series.shape[1] - horizon + 1) if observed[origin:origin + horizon].all()]
    if len(origins) < count:
        raise ValueError(f"Need {count} distinct origins with {horizon} observed future labels after context {context}; found {len(origins)}")
    # Integer spacing includes the earliest/latest eligible origins and is stable.
    chosen = [origins[(index * (len(origins) - 1)) // max(1, count - 1)] for index in range(count)]
    cases = []
    for window, origin in enumerate(chosen):
        channels = [index % len(prepare_data.COLUMNS) for index in range(batch_size)]
        inputs = [{"target": series[channel, origin - context:origin].tolist(),
                   "item_id": f"mp_1/{prepare_data.COLUMNS[channel][0]}/origin-{origin}/row-{row}"}
                  for row, channel in enumerate(channels)]
        cases.append(validate_case({
            "name": f"assetops-c{context}-h{horizon}-b{batch_size}-window-{window}",
            "data": {"inputs": inputs, "parameters": {"prediction_length": horizon, "quantile_levels": list(QUANTILES)}},
            "evaluation": {"targets": [series[channel, origin:origin + horizon].tolist() for channel in channels],
                           "observed": [[True] * horizon for _ in channels], "seasonality": 24},
            "provenance": {**provenance, "origin": origin, "forecast_start": timestamps[origin],
                           "history_start": timestamps[origin - context], "channel_rows": channels,
                           "past_imputed_positions": [index for index in range(origin - context, origin) if not observed[index]],
                           "repeated_windows": max(0, batch_size - 7),
                           "performance_only_repetitions": batch_size > 7},
        }))
    return cases


NON_OVERLAPPING = "non-overlapping"
# One HTTP request carries at most this many windows. Moirai admits 8 rows per
# request (TimesFM and Toto admit 16), so one manifest serves all three adapters.
SIGNAL_ROWS_PER_CASE = 8


def select_origins(origins, windows, horizon):
    """Choose forecast origins from the sorted eligible ``origins``.

    ``windows=NON_OVERLAPPING`` takes every origin whose horizon starts at least
    ``horizon`` slots after the previously chosen one, greedily from the earliest
    eligible origin. Forecast horizons never overlap; histories may. An integer
    keeps the original policy: that many evenly spaced origins, including the
    earliest and latest (the November 14 baseline used 5).
    """
    if windows == NON_OVERLAPPING:
        chosen = []
        for origin in origins:
            if not chosen or origin >= chosen[-1] + horizon:
                chosen.append(origin)
        return chosen
    _positive_integer(windows, "windows")
    if len(origins) < windows:
        return []
    return [origins[(index * (len(origins) - 1)) // max(1, windows - 1)] for index in range(windows)]


def signal_cases(data_dir, names=None, *, context=512, horizon=96, windows=NON_OVERLAPPING,
                 rows_per_case=SIGNAL_ROWS_PER_CASE):
    """Build accuracy cases per pinned signal from ``assetops_sources.json``.

    Rows are windows of shape [context] from one signal whose next ``horizon``
    slots were all actually recorded, chosen by :func:`select_origins`.
    Histories may contain slots causally forward-filled from the signal's own
    earlier readings; those positions are recorded and are never scored. Item
    IDs are ``<signal>/origin-<slot>``, so tables group a signal's windows even
    when more than ``rows_per_case`` windows are split across several cases
    (``...-part-<n>``; a signal that fits one case keeps its original name).
    Reads files already placed by
    ``python -m benchmarks.prepare_data --assetops-signals``.
    """
    for value, label in [(context, "context"), (horizon, "horizon"), (rows_per_case, "rows_per_case")]:
        _positive_integer(value, label)
    if windows != NON_OVERLAPPING:
        _positive_integer(windows, "windows")
    if horizon > 1024:
        raise ValueError("prediction_length must be <= 1024")
    names = list(prepare_data.SOURCES["signals"]) if names is None else list(names)
    cases = []
    for name in names:
        signal = prepare_data.load_signal(data_dir, name)
        values, observed = signal["values"], signal["observed"]
        first = observed.index(True)
        filled, last = [], None
        for value in values:
            last = value if value is not None else last
            filled.append(last)
        origins = [origin for origin in range(first + context, len(values) - horizon + 1)
                   if all(observed[origin:origin + horizon])]
        chosen = select_origins(origins, windows, horizon)
        if not chosen:
            raise ValueError(f"{name}: need {windows if windows != NON_OVERLAPPING else 1} origins with {horizon} "
                             f"recorded future values after context {context}; found {len(origins)}")
        parts = [chosen[start:start + rows_per_case] for start in range(0, len(chosen), rows_per_case)]
        for part, part_origins in enumerate(parts, start=1):
            imputed_positions = [[index for index in range(origin - context, origin) if not observed[index]]
                                 for origin in part_origins]
            base = f"signal-{name.replace('/', '-')}-c{context}-h{horizon}"
            cases.append(validate_case({
                "name": base if len(parts) == 1 else f"{base}-part-{part}",
                "data": {"inputs": [{"target": filled[origin - context:origin], "item_id": f"{name}/origin-{origin}"}
                                    for origin in part_origins],
                         "parameters": {"prediction_length": horizon, "quantile_levels": list(QUANTILES)}},
                "evaluation": {"targets": [values[origin:origin + horizon] for origin in part_origins],
                               "observed": [[True] * horizon for _ in part_origins],
                               "seasonality": signal["seasonality"]},
                "provenance": {**signal["provenance"], "origins": part_origins,
                               "forecast_starts": [signal["axis"][origin] for origin in part_origins],
                               "past_imputed_positions": imputed_positions,
                               "past_imputed_counts": [len(positions) for positions in imputed_positions],
                               "window_policy": windows if windows == NON_OVERLAPPING else f"evenly-spaced-{windows}",
                               "signal_windows": len(chosen), "eligible_origins": len(origins),
                               "part": part, "parts": len(parts),
                               "history_policy": "Recorded readings; missing slots causally forward-filled from earlier readings only",
                               "truth_policy": "Only windows whose whole horizon was recorded; no filled value is scored",
                               "sampling_unit": "Windows from one asset's signal; correlated, not independent assets"},
            }))
    return cases


def seasonal_naive_forecasts(case):
    """Return [series, nine quantiles, horizon] repeating the last season."""
    import numpy as np

    validate_case(case)
    horizon = case["data"]["parameters"]["prediction_length"]
    seasonality = case["evaluation"]["seasonality"]
    rows = []
    for item in case["data"]["inputs"]:
        past = item["target"]
        if len(past) < seasonality:
            raise ValueError("Seasonal naive needs at least one complete past season")
        rows.append([[past[-seasonality + (step % seasonality)] for step in range(horizon)]] * len(QUANTILES))
    return np.asarray(rows, dtype=float)


def mean_history_forecasts(case):
    """Return [series, nine quantiles, horizon] repeating the mean of recorded history.

    Causally forward-filled slots (see :func:`history_observed`) are excluded, so
    the baseline uses only readings that were actually recorded. Every step and
    quantile gets the same value: a point baseline whose SQL equals its MASE.
    Cases without imputation provenance use their whole history.
    """
    import numpy as np

    validate_case(case)
    horizon = case["data"]["parameters"]["prediction_length"]
    masks = history_observed(case)
    rows = []
    for row, item in enumerate(case["data"]["inputs"]):
        past = np.asarray(item["target"], dtype=float)
        recorded = past if masks is None else past[np.asarray(masks[row], dtype=bool)]
        if not recorded.size:
            raise ValueError(f"{case['name']}: row {row} has no recorded history for the mean baseline")
        rows.append(np.full((len(QUANTILES), horizon), recorded.mean()))
    return np.asarray(rows, dtype=float)


SCALES = ("all", "observed")


def history_observed(case):
    """Per-row history masks: True where a value was recorded, False where filled.

    Read from provenance: signal cases (``origins`` with per-row absolute
    ``past_imputed_positions``), MetroPT fixture cases (one ``origin`` shared by
    all rows) and fev cases (``series[i].past_imputed_offsets`` within each
    history). Returns None for a case that does not record its imputation.
    """
    provenance = case.get("provenance") or {}
    lengths = [len(item["target"]) for item in case["data"]["inputs"]]
    if "origins" in provenance and "past_imputed_positions" in provenance:
        rows = [(origin - length, positions) for origin, length, positions
                in zip(provenance["origins"], lengths, provenance["past_imputed_positions"], strict=True)]
    elif "origin" in provenance and "past_imputed_positions" in provenance:
        rows = [(provenance["origin"] - length, provenance["past_imputed_positions"]) for length in lengths]
    elif isinstance(provenance.get("series"), list) and all("past_imputed_offsets" in row for row in provenance["series"]):
        rows = [(0, row["past_imputed_offsets"]) for row in provenance["series"]]
    else:
        return None
    masks = []
    for (start, positions), length in zip(rows, lengths, strict=True):
        mask = [True] * length
        for position in positions:
            if not 0 <= position - start < length:
                raise ValueError(f"{case['name']}: imputed position {position} is outside its history")
            mask[position - start] = False
        masks.append(mask)
    return masks


def seasonal_scale(past, seasonality, observed=None):
    """Mean absolute lag-``seasonality`` difference in a history, and its pair count.

    With ``observed``, only pairs whose two values were both recorded count, so
    forward-filled runs cannot shrink the scale. Returns (None, 0) without pairs.
    """
    pairs = [abs(past[index] - past[index - seasonality]) for index in range(seasonality, len(past))
             if observed is None or (observed[index] and observed[index - seasonality])]
    return (sum(pairs) / len(pairs), len(pairs)) if pairs else (None, 0)


def score_forecasts(case, values, *, scale="all"):
    """Score [N, 9, H] forecasts against observed future labels, using past only.

    MAE uses the median. MASE divides median MAE by mean absolute seasonal
    differences in the supplied past. ``scale="all"`` (default) uses every pair,
    including documented causal fills; ``scale="observed"`` uses only pairs whose
    two values were recorded (see ``history_observed``). SQL averages twice the
    pinball loss over all nine quantiles, divided by the same scale. Aggregates
    are equal-weight means over eligible series. Undefined scales (zero, or no
    usable pairs) are excluded explicitly, never converted to zero or JSON NaN.
    """
    import numpy as np

    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    validate_case(case)
    masks = history_observed(case)
    if scale == "observed" and masks is None:
        raise ValueError(f"{case['name']}: observed-pair scaling needs recorded history imputation")
    inputs, evaluation = case["data"]["inputs"], case["evaluation"]
    horizon = case["data"]["parameters"]["prediction_length"]
    forecast = np.asarray(values, dtype=float)
    if forecast.shape != (len(inputs), 9, horizon) or not np.isfinite(forecast).all():
        raise ValueError(f"Forecasts must be finite with shape {(len(inputs), 9, horizon)}")
    per_series = []
    for row, item in enumerate(inputs):
        observed = np.asarray(evaluation["observed"][row], dtype=bool)
        truth = np.asarray([value if flag else 0.0 for value, flag in zip(evaluation["targets"][row], observed)], dtype=float)
        past = np.asarray(item["target"], dtype=float)
        seasonality = evaluation["seasonality"]
        differences = np.abs(past[seasonality:] - past[:-seasonality])
        if scale == "observed":
            recorded = np.asarray(masks[row], dtype=bool)
            differences = differences[recorded[seasonality:] & recorded[:-seasonality]]
        series_scale = float(differences.mean()) if differences.size else None
        status = "insufficient_history" if series_scale is None else "zero" if series_scale == 0 else "valid"
        mae = mase = sql = None
        if observed.any():
            mae = float(np.abs(forecast[row, 4, observed] - truth[observed]).mean())
            if status == "valid":
                errors = truth[None, observed] - forecast[row][:, observed]
                quantiles = np.asarray(QUANTILES)[:, None]
                mase = mae / series_scale
                sql = float((2 * np.maximum(quantiles * errors, (quantiles - 1) * errors)).mean() / series_scale)
        per_series.append({"item_id": item["item_id"], "MAE": mae, "MASE": mase, "SQL": sql,
                           "seasonal_scale": series_scale, "scale_status": status,
                           "scale_pairs": int(differences.size),
                           "filled_history": None if masks is None else masks[row].count(False),
                           "observed_targets": int(observed.sum())})
    def average(metric):
        eligible = [row[metric] for row in per_series if row[metric] is not None]
        return sum(eligible) / len(eligible) if eligible else None
    result = {"MAE": average("MAE"), "MASE": average("MASE"), "SQL": average("SQL"), "scale": scale,
            "counts": {"series": len(inputs), "scored_series": sum(row["MAE"] is not None for row in per_series),
                       "observed_targets": sum(row["observed_targets"] for row in per_series),
                       "valid_scaled_series": sum(row["MASE"] is not None for row in per_series),
                       "zero_scale_series": sum(row["scale_status"] == "zero" for row in per_series),
                       "insufficient_history_series": sum(row["scale_status"] == "insufficient_history" for row in per_series),
                       "unobserved_series": sum(row["observed_targets"] == 0 for row in per_series)},
            "per_series": per_series}
    # Extreme but finite inputs can still overflow arithmetic. Do not serialize
    # nonfinite scores or let an infinite denominator silently report success.
    json.dumps(result, allow_nan=False)
    return result


def _causal_fill(values):
    """Fill missing past values forward without ever inspecting future labels."""
    filled, positions = [], []
    for index, value in enumerate(values):
        if _finite(value):
            filled.append(float(value))
        elif filled:
            filled.append(filled[-1])
            positions.append(index)
        else:
            raise ValueError("Cannot causally fill a history that starts with a missing observation")
    return filled, positions


def fev_cases(cache_dir, config_path=None):
    """Download frozen public data and materialize three custom fev smoke tasks.

    This intentionally covers only univariate, target-only slices, at most eight
    compatible series and two rolling windows per task. It is not fev-bench.
    fev 0.10 has no HF revision field: pinned parquet files are selected and saved
    locally, then passed through the public fev.Task/EvaluationWindow API.
    """
    import dataclasses
    from importlib.metadata import version

    import datasets
    import fev
    from huggingface_hub import HfApi, hf_hub_download

    if version("fev") != FEV_VERSION:
        raise ValueError(f"Evaluation preparation requires fev=={FEV_VERSION}")
    config = json.loads(Path(config_path or Path(__file__).with_name("fev_smoke.json")).read_text())
    if config.get("fev_version") != FEV_VERSION or config.get("quantile_levels") != QUANTILES:
        raise ValueError("fev smoke config must match the pinned version and all nine quantiles")
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    revision = config["dataset_revision"]
    info = HfApi().dataset_info(config["dataset_path"], revision=revision)
    if info.sha != revision or len(revision) != 40:
        raise ValueError("fev dataset revision must resolve to the recorded full commit SHA")
    cases = []
    for definition in config["tasks"]:
        source_file = definition["dataset_config"] + "/train-00000-of-00001.parquet"
        parquet = hf_hub_download(config["dataset_path"], source_file, repo_type="dataset", revision=revision, cache_dir=root / "hub")
        source_file_sha256 = hashlib.sha256(Path(parquet).read_bytes()).hexdigest()
        source = datasets.load_dataset("parquet", data_files=parquet, split="train", cache_dir=str(root / "datasets"))
        context, horizon = definition["context"], definition["horizon"]
        target_column = definition["target"]
        selected, source_indices = [], []
        for index, row in enumerate(source):
            target = row[target_column]
            if len(target) < context + 2 * horizon:
                continue
            eligible = True
            for offset in [-2 * horizon, -horizon]:
                origin = len(target) + offset
                try:
                    _causal_fill(target[origin - context:origin])
                except ValueError:
                    eligible = False
                if not all(_finite(value) for value in target[origin:origin + horizon]):
                    eligible = False
            if eligible:
                selected.append({"id": str(row["id"]), "timestamp": row["timestamp"], target_column: target})
                source_indices.append(index)
                if len(selected) == definition["max_series"]:
                    break
        if not selected:
            raise ValueError(f"No compatible observed-label series in {definition['dataset_config']}")
        local = root / definition["dataset_config"] / "data.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_list(selected).to_parquet(local)
        task = fev.Task(dataset_path=str(local), horizon=horizon, num_windows=2,
                        window_step_size=horizon, min_context_length=context, max_context_length=context,
                        seasonality=definition["seasonality"], eval_metric="SQL", extra_metrics=["MASE", "MAE"],
                        quantile_levels=list(QUANTILES), target=target_column, task_name=definition["dataset_config"])
        task_parameters = dataclasses.asdict(task)
        # The local cache path is an implementation detail, not the source identity.
        task_parameters["dataset_path"] = "<pinned selected-series parquet>"
        for window_index, window in enumerate(task.iter_windows(num_proc=1)):
            past_data, _ = window.get_input_data()
            truth = window.get_ground_truth()
            inputs, targets, masks, series_provenance = [], [], [], []
            for past, future, source_index, original in zip(past_data, truth, source_indices, selected):
                if str(past["id"]) != str(future["id"]):
                    raise ValueError("fev input/ground-truth item order differs")
                history, imputed = _causal_fill(past[target_column])
                target = [float(value) for value in future[target_column]]
                inputs.append({"target": history, "item_id": str(past["id"])})
                targets.append(target)
                masks.append([True] * horizon)
                series_provenance.append({"item_id": str(past["id"]), "source_row": source_index,
                                          "origin": len(original[target_column]) + window.cutoff,
                                          "history_start": str(past["timestamp"][0]),
                                          "history_end": str(past["timestamp"][-1]),
                                          "forecast_start": str(future["timestamp"][0]),
                                          "past_imputed_offsets": imputed})
            cases.append(validate_case({
                "name": f"fev-smoke-{definition['dataset_config']}-window-{window_index}",
                "data": {"inputs": inputs, "parameters": {"prediction_length": horizon, "quantile_levels": list(QUANTILES)}},
                "evaluation": {"targets": targets, "observed": masks, "seasonality": task.seasonality},
                "provenance": {"dataset": "Custom univariate fev smoke subset (not fev-bench)",
                               "dataset_path": config["dataset_path"], "dataset_config": definition["dataset_config"],
                               "dataset_revision": revision, "source_file": source_file,
                               "source_file_sha256": source_file_sha256,
                               "source_target": target_column,
                               "excluded_source_columns": [column for column in source.column_names if column not in {"id", "timestamp", target_column}],
                               "fev_version": FEV_VERSION, "datasets_version": version("datasets"),
                               "task": task_parameters, "window_index": window_index, "cutoff": window.cutoff,
                               "selection": "First compatible series in pinned source row order; no missing future labels",
                               "history_policy": "At most configured context; missing past causally forward-filled",
                               "series": series_provenance},
            }))
    return cases


def _window_policy(value):
    if value == NON_OVERLAPPING:
        return value
    try:
        return _positive_integer(int(value), "windows")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected '{NON_OVERLAPPING}' or a positive integer, got {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Prepare observed-label accuracy workloads; no model required")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True, help="New JSON manifest (existing files are rejected)")
    prepare.add_argument("--fev", action="store_true", help="Also download and prepare the three pinned fev smoke tasks")
    prepare.add_argument("--cache-dir", type=Path, help="fev download/cache directory (default: beside output)")
    signals = prepare.add_argument_group(
        "AssetOps signals", "Cases from the named signals in benchmarks/assetops_sources.json. "
        "These replace the default MetroPT fixture windows; --fev still adds its tasks.")
    signals.add_argument("--assetops-signals", action="store_true", help="Use every pinned signal")
    signals.add_argument("--signal", action="append", metavar="NAME",
                         choices=sorted(prepare_data.SOURCES["signals"]), help="Use this signal (repeatable)")
    signals.add_argument("--signal-context", type=int, default=512)
    signals.add_argument("--signal-horizon", type=int, default=96)
    signals.add_argument("--signal-windows", type=_window_policy, default=NON_OVERLAPPING,
                         help=f"'{NON_OVERLAPPING}' (default: every window whose horizon does not overlap another) "
                              "or an integer count of evenly spaced origins (5 reproduces the Nov 14 baseline)")
    signals.add_argument("--signal-rows-per-case", type=int, default=SIGNAL_ROWS_PER_CASE,
                         help="Windows per request; more windows are split into several cases per signal")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    names = args.signal or (list(prepare_data.SOURCES["signals"]) if args.assetops_signals else None)
    if names is None:
        cases = assetops_cases(args.data_dir, count=5, batch_size=7)
    else:
        cases = signal_cases(args.data_dir, names, context=args.signal_context,
                             horizon=args.signal_horizon, windows=args.signal_windows,
                             rows_per_case=args.signal_rows_per_case)
    if args.fev:
        cases.extend(fev_cases(args.cache_dir or args.output.parent / "fev-cache"))
    manifest = {"cases": cases, "provenance": {
        "purpose": "Small real-data adapter quality checks, not a full benchmark or model ranking",
        "quantile_levels": list(QUANTILES), "fev_custom_smoke": args.fev,
        "metrics": "Median MAE/MASE; SQL averages twice pinball loss across nine quantiles; past-only seasonal scale",
        **({"assetops_windows": 5} if names is None else {
            "assetops_signals": names, "signal_context": args.signal_context,
            "signal_horizon": args.signal_horizon, "signal_windows": args.signal_windows,
            "signal_rows_per_case": args.signal_rows_per_case,
            "signal_window_counts": {name: max(case["provenance"]["signal_windows"] for case in cases
                                               if case["data"]["inputs"][0]["item_id"].startswith(f"{name}/origin-"))
                                     for name in names},
            "sources_commit": prepare_data.COMMIT}),
    }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(cases), "fev_custom_smoke": args.fev}))


if __name__ == "__main__":
    main()
