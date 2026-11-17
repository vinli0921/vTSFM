#!/usr/bin/env python3
"""Prepare pinned AssetOpsBench numeric fixtures without generating observations.

Standard library only. ``assetops_sources.json`` pins every AssetOpsBench sensor
file this repository uses, plus the named signals built from them. The CLI
prepares the MetroPT fixture used by the integration, performance and default
accuracy checks. ``--assetops-signals`` also fetches files used by the named
signals. Each source is cached in ``<output>/source/`` and verified by SHA256.

The fixture is a NumPy-compatible data.npz containing ``series`` with shape
[7, 5116], dtype float32, and a provenance manifest. The source has 560 wholly
empty hourly rows. They are causally forward-filled from the last available
observation, with all imputed indices documented. The channels share one asset
and time axis; overlapping windows are correlated.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import urllib.request
import zipfile


SOURCES_FILE = Path(__file__).with_name("assetops_sources.json")


def load_sources(path: Path = SOURCES_FILE) -> dict:
    """Read and check the pinned AssetOpsBench file and signal list."""
    sources = json.loads(Path(path).read_text())
    commit, files, signals = sources.get("commit"), sources.get("files"), sources.get("signals")
    if not isinstance(commit, str) or len(commit) != 40 or not isinstance(sources.get("directory"), str):
        raise ValueError("AssetOps sources need a full commit SHA and a source directory")
    if not isinstance(files, dict) or not files:
        raise ValueError("AssetOps sources need a nonempty files mapping")
    for name, entry in files.items():
        if "/" in name or not name.endswith(".json") or len(str(entry.get("sha256", ""))) != 64:
            raise ValueError(f"Pinned AssetOps file {name!r} needs a plain JSON name and SHA256")
    for name, signal in (signals or {}).items():
        axis = signal.get("axis", {})
        if (signal.get("file") not in files or not isinstance(signal.get("column"), str)
                or not isinstance(axis.get("field"), str) or ("step_seconds" in axis) == ("step" in axis)
                or type(signal.get("seasonality")) is not int or signal["seasonality"] < 1):
            raise ValueError(f"Signal {name!r} needs a pinned file, column, axis rule and seasonality")
    return sources


SOURCES = load_sources()
COMMIT = SOURCES["commit"]
SOURCE_PATH = f"{SOURCES['directory']}/metro_pump_1.json"
SOURCE_URL = f"https://raw.githubusercontent.com/{SOURCES['repository']}/{COMMIT}/{SOURCE_PATH}"
SOURCE_SHA256 = SOURCES["files"]["metro_pump_1.json"]["sha256"]
COLUMNS = (
    ("Compressor_Pressure_bar", "bar", "TP2"),
    ("Pneumatic_Panel_Pressure_bar", "bar", "TP3"),
    ("Cyclone_Filter_Drop_Pressure_bar", "bar", "H1"),
    ("Tower_Discharge_Pressure_Drop_bar", "bar", "DV_pressure"),
    ("Reservoir_Pressure_bar", "bar", "Reservoirs"),
    ("Oil_Temperature_C", "degC", "Oil_temperature"),
    ("Motor_Current_A", "A", "Motor_current"),
)
CONTEXTS = (128, 512, 2048, 4096)
HORIZON = 96


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def fetch_source(source_file: Path | None, cache_path: Path, url: str | None = None,
                 expected_sha256: str | None = None) -> bytes:
    """Read a local copy, the cache, or the pinned URL; always verify SHA256.

    The URL and checksum default to the MetroPT fixture source.
    """
    url = SOURCE_URL if url is None else url
    expected_sha256 = SOURCE_SHA256 if expected_sha256 is None else expected_sha256
    if source_file is not None:
        content = source_file.read_bytes()
    elif cache_path.exists():
        content = cache_path.read_bytes()
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AssetOpsBench-numeric-benchmark/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    actual = sha256(content)
    if actual != expected_sha256:
        raise ValueError(f"Source checksum mismatch: expected {expected_sha256}, got {actual}")
    return content


def source_url(name: str, sources: dict | None = None) -> str:
    sources = SOURCES if sources is None else sources
    return (f"https://raw.githubusercontent.com/{sources['repository']}/{sources['commit']}/"
            f"{sources['directory']}/{name}")


def fetch_sources(output_dir: Path, names=None, *, checkout: Path | None = None,
                  sources: dict | None = None) -> dict[str, str]:
    """Place every pinned file (or ``names``) in ``output_dir/source/``.

    Files already present are verified and kept, so repeated runs download
    nothing. ``checkout`` copies from a local AssetOpsBench checkout instead of
    downloading; its files must still match the pins. A present file with the
    wrong checksum is an error rather than being silently replaced.
    """
    sources = SOURCES if sources is None else sources
    names = list(sources["files"]) if names is None else list(names)
    unknown = sorted(set(names) - set(sources["files"]))
    if unknown:
        raise ValueError(f"Unpinned AssetOps files: {', '.join(unknown)}")
    cache_dir = Path(output_dir) / "source"
    cache_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for name in names:
        expected = sources["files"][name]["sha256"]
        cache_path = cache_dir / name
        if cache_path.exists():
            if sha256(cache_path.read_bytes()) != expected:
                raise ValueError(f"{cache_path} does not match its pinned SHA256; delete it to fetch again")
            status[name] = "present"
            continue
        local = None if checkout is None else Path(checkout) / sources["directory"] / name
        content = fetch_source(local, cache_path, source_url(name, sources), expected)
        atomic_write(cache_path, content)
        status[name] = "copied" if local is not None else "downloaded"
    return status


def _numeric(value, name, index):
    """Return a finite reading, or None for an absent/empty one."""
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Non-numeric or nonfinite {name} at source row {index}")
    return float(value)


def load_signal(data_dir: Path, name: str, *, sources: dict | None = None) -> dict:
    """Load one pinned signal as a regular series with an observed mask.

    Run this module's CLI with ``--assetops-signals`` to prepare the local
    sources. Each timestamp goes to the nearest slot on a ``step_seconds`` grid
    and must lie within a quarter step of it. Counter axes
    (e.g. hydraulic ``cycle``) use one slot per integer step. Slots without a
    reading are ``None`` with ``observed`` false; two readings in one slot are
    an error. Returns ``values``, ``observed``, ``axis`` labels, ``seasonality``
    and ``provenance``.
    """
    sources = SOURCES if sources is None else sources
    if name not in sources.get("signals", {}):
        raise ValueError(f"Unknown AssetOps signal {name!r}")
    signal = sources["signals"][name]
    file_name, column, axis = signal["file"], signal["column"], signal["axis"]
    path = Path(data_dir) / "source" / file_name
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run python -m benchmarks.prepare_data --assetops-signals "
                         "with --output set to your data directory.")
    content = path.read_bytes()
    if sha256(content) != sources["files"][file_name]["sha256"]:
        raise ValueError(f"{path} does not match its pinned SHA256")
    rows = json.loads(content)
    if not isinstance(rows, list):
        raise ValueError(f"{file_name} must contain a list of rows")

    readings, absent = [], 0
    for index, row in enumerate(rows):
        if "asset_id" in signal and row.get("asset_id") != signal["asset_id"]:
            raise ValueError(f"Unexpected asset {row.get('asset_id')!r} at {file_name} row {index}")
        value = _numeric(row.get(column), column, index)
        if value is None:
            absent += 1
            continue
        key = row.get(axis["field"])
        if "step_seconds" in axis:
            try:
                key = datetime.fromisoformat(key)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {axis['field']} at {file_name} row {index}: {key!r}") from error
        elif type(key) is not int:
            raise ValueError(f"Non-integer {axis['field']} at {file_name} row {index}")
        readings.append((key, index, value))
    if not readings:
        raise ValueError(f"{name} has no readings")
    readings.sort(key=lambda reading: (reading[0], reading[1]))

    first = readings[0][0]
    slots, max_offset = {}, 0.
    if "step_seconds" in axis:
        # Align slots to clock multiples of the step, so a jittered first
        # reading does not shift every later slot.
        step = axis["step_seconds"]
        epoch = datetime(1970, 1, 1, tzinfo=first.tzinfo)
        base = round((first - epoch).total_seconds() / step)
    for key, index, value in readings:
        if "step_seconds" in axis:
            elapsed = (key - epoch).total_seconds()
            slot = round(elapsed / step)
            offset = abs(elapsed - slot * step)
            if offset > step / 4:
                raise ValueError(f"{file_name} row {index} is {offset:.0f}s from the {step}s grid")
            max_offset = max(max_offset, offset)
            slot -= base
        else:
            delta = key - first
            if delta % axis["step"]:
                raise ValueError(f"{file_name} row {index} is off the {axis['field']} step")
            slot = delta // axis["step"]
        if slot in slots:
            raise ValueError(f"Two {column} readings share slot {slot} in {file_name}")
        slots[slot] = value
    count = max(slots) + 1
    values = [slots.get(slot) for slot in range(count)]
    observed = [value is not None for value in values]
    if "step_seconds" in axis:
        labels = [(epoch + timedelta(seconds=(base + slot) * step)).isoformat() for slot in range(count)]
    else:
        labels = [str(first + slot * axis["step"]) for slot in range(count)]
    missing = [slot for slot, flag in enumerate(observed) if not flag]
    provenance = {
        "signal": name, "repository": sources["repository"], "commit": sources["commit"],
        "path": f"{sources['directory']}/{file_name}", "sha256": sources["files"][file_name]["sha256"],
        "column": column, "asset_id": signal.get("asset_id"), "axis": axis,
        "source_rows": len(rows), "rows_without_value": absent, "readings": len(readings),
        "slots": count, "missing_slots": len(missing), "missing_slot_indices": missing,
        "max_grid_offset_seconds": max_offset if "step_seconds" in axis else None,
        "first": labels[0], "last": labels[-1], "notes": signal.get("notes"),
    }
    return {"values": values, "observed": observed, "axis": labels,
            "seasonality": signal["seasonality"], "provenance": provenance}


def numpy_array_bytes(shape: tuple[int, int], values: list[list[float]]) -> bytes:
    """Encode a little-endian float32 NPY v1 array, with no NumPy dependency."""
    header_text = repr({"descr": "<f4", "fortran_order": False, "shape": shape})
    # A v1 preamble occupies 10 bytes; the complete header aligns to 64 bytes.
    encoded = header_text.encode("latin1")
    padding = (-(10 + len(encoded) + 1)) % 64
    header = encoded + b" " * padding + b"\n"
    body = array("f", (value for channel in values for value in channel))
    if body.itemsize != 4:
        raise RuntimeError("This platform does not provide 32-bit array('f') values")
    if sys.byteorder != "little":
        body.byteswap()
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body.tobytes()


def prepare(output_dir: Path, source_file: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "source"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "metro_pump_1.json"
    content = fetch_source(source_file, cache_path)
    source_rows = json.loads(content)
    if not isinstance(source_rows, list) or len(source_rows) != 5116:
        raise ValueError("Expected the pinned fixture's 5,116 rows")
    if {row.get("asset_id") for row in source_rows} != {"mp_1"}:
        raise ValueError("Expected only asset mp_1")
    source_timestamps = [row["timestamp"] for row in source_rows]
    parsed = [datetime.fromisoformat(value) for value in source_timestamps]
    deltas = Counter((end - start).total_seconds() for start, end in zip(parsed, parsed[1:]))
    if deltas != Counter({3600.0: len(source_rows) - 1}):
        raise ValueError(f"Expected chronological hourly observations; intervals={dict(deltas)}")

    rows = []
    source_indices = []
    missing_indices = []
    dropped_leading_indices = []
    last_observed = None
    for index, row in enumerate(source_rows):
        observed = [row.get(name) for name, _, _ in COLUMNS]
        if all(value == "" or value is None for value in observed):
            missing_indices.append(index)
            if last_observed is None:
                dropped_leading_indices.append(index)
                continue
            filled = dict(row)
            for name, _, _ in COLUMNS:
                filled[name] = last_observed[name]
            rows.append(filled)
        else:
            for (name, _, _), value in zip(COLUMNS, observed):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Partial, non-numeric, or nonfinite {name} at source row {index}")
            rows.append(row)
            last_observed = row
        source_indices.append(index)
    if len(missing_indices) != 560 or len(rows) != 5116:
        raise ValueError("Expected 560 wholly missing rows and 5,116 rows after causal forward-fill")
    imputed_indices = [index for index in missing_indices if index not in dropped_leading_indices]
    timestamps = [row["timestamp"] for row in rows]
    retained_times = [datetime.fromisoformat(value) for value in timestamps]
    retained_deltas = Counter((end - start).total_seconds() for start, end in zip(retained_times, retained_times[1:]))

    values = []
    column_metadata = []
    for name, unit, upstream_name in COLUMNS:
        channel = []
        for index, row in enumerate(rows):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric or missing {name} at source row {index}")
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {name} at source row {index}")
            channel.append(float(value))
        values.append(channel)
        column_metadata.append({
            "array_row": len(values) - 1,
            "name": name,
            "unit": unit,
            "upstream_sensor_correspondence": upstream_name,
            "upstream_correspondence_status": "semantic correspondence from public sensor descriptions",
            "source_json_column": name,
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "source_rows": len(source_rows),
            "source_missing_values": len(missing_indices),
            "observed_source_values": len(source_rows) - len(missing_indices),
            "imputed_values": len(imputed_indices),
            "output_values": len(channel),
            "finite_output_values": len(channel),
            "minimum_observed_source_value": min(channel),
            "maximum_observed_source_value": max(channel),
        })

    shape = (len(COLUMNS), len(rows))
    npy = numpy_array_bytes(shape, values)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        info = zipfile.ZipInfo("series.npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        target.writestr(info, npy)
    npz = archive.getvalue()

    manifest = {
        "dataset_name": "AssetOpsBench hourly MetroPT compressor fixture with causal forward-fill",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "IBM/AssetOpsBench",
            "commit": COMMIT,
            "path": SOURCE_PATH,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "bytes": len(content),
            "local_copy": "source/metro_pump_1.json",
            "repository_license": "Apache-2.0",
            "repository_license_url": f"https://github.com/IBM/AssetOpsBench/blob/{COMMIT}/LICENSE",
            "upstream_dataset": "MetroPT-3",
            "upstream_doi": "https://doi.org/10.24432/C5VW3R",
            "upstream_license": "CC-BY-4.0",
            "upstream_license_source": "https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B",
            "upstream_citation": "Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). MetroPT-3 Dataset. UCI Machine Learning Repository.",
            "upstream_relationship_evidence": "https://arxiv.org/html/2506.03828v3#S6",
            "provenance_limit": "The AssetOpsBench paper identifies MetroPT-3 as compressor data. The exact raw-to-hourly fixture aggregation recipe is not documented in the inspected repository; no new aggregation is performed here.",
        },
        "array": {"file": "data.npz", "key": "series", "shape": list(shape), "dtype": "float32", "sha256": sha256(npz), "series_npy_sha256": sha256(npy)},
        "asset_id": "mp_1",
        "time_axis": {
            "first": timestamps[0], "last": timestamps[-1],
            "observations": len(rows), "source_grid_cadence_seconds": 3600,
            "output_sampling": "hourly grid; wholly missing sensor rows causally forward-filled",
            "output_delta_seconds_counts": {str(int(delta)): count for delta, count in sorted(retained_deltas.items())},
            "maximum_output_gap_seconds": max(retained_deltas),
            "longest_contiguous_hourly_observed_run": 224,
            "timezone": "not specified in source; timestamps retained as provided",
            "timestamp_sidecar": "timestamps.json",
            "source_row_indices_sidecar": "source_row_indices.json",
        },
        "columns": column_metadata,
        "transformations": ["Select seven analog sensor columns", "Causally forward-fill 560 wholly missing hourly rows from the most recent observed values, without using future values", "Retain chronological hourly order and all 4,556 observed numeric values per selected channel", "Cast numeric values to IEEE 754 float32"],
        "synthetically_generated_observations": 0,
        "imputed_sensor_values": len(imputed_indices) * len(COLUMNS),
        "dropped_rows": len(dropped_leading_indices),
        "dropped_source_row_indices": dropped_leading_indices,
        "imputation": {
            "method": "causal forward-fill from last observed sensor values; no backward-fill",
            "missing_rows_all_selected_channels": len(missing_indices),
            "imputed_rows": len(imputed_indices),
            "source_row_indices": imputed_indices,
            "timestamps": [source_timestamps[index] for index in imputed_indices],
            "observed_source_values_preserved_before_float32_cast": True,
        },
        "normalization": None,
        "benchmark": {
            "recommended_context_lengths": list(CONTEXTS),
            "forecast_horizon": HORIZON,
            "maximum_context_with_holdout": len(rows) - HORIZON,
            "distinct_channel_start_windows": {
                str(context): len(COLUMNS) * max(0, len(rows) - context - HORIZON + 1)
                for context in CONTEXTS
            },
            "window_semantics": "Different sensor/start windows, potentially overlapping, from one real compressor asset. These windows are correlated and are not distinct independent assets. Context/horizon lengths count hourly grid positions, including documented forward-filled values. Reserve the next 96 grid values for each holdout. This dataset is for performance measurement; no forecast-accuracy claims are made.",
            "cross_learning": False,
            "unsupported_context_lengths": [8192],
        },
    }
    atomic_write(cache_path, content)
    atomic_write(output_dir / "data.npz", npz)
    atomic_write(output_dir / "timestamps.json", (json.dumps(timestamps) + "\n").encode())
    atomic_write(output_dir / "source_row_indices.json", (json.dumps(source_indices) + "\n").encode())
    atomic_write(output_dir / "data_manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=Path("data"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--source-file", type=Path, help="Prepare only MetroPT from a verified local copy")
    selection.add_argument("--assetops-signals", action="store_true", help="Also fetch files used by the pinned signals")
    parser.add_argument("--assetops-checkout", type=Path,
                        help="Copy requested files from a local AssetOpsBench checkout instead of downloading")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if args.source_file is not None:
        manifest = prepare(output, args.source_file)
        fetched = {"metro_pump_1.json": "copied"}
    else:
        names = {"metro_pump_1.json"}
        if args.assetops_signals:
            names.update(signal["file"] for signal in SOURCES["signals"].values())
        fetched = fetch_sources(output, sorted(names), checkout=args.assetops_checkout)
        manifest = prepare(output)
    print(json.dumps({"output_dir": str(output.resolve()), "sources": fetched,
                      "array": manifest["array"], "benchmark": manifest["benchmark"]}, indent=2))


if __name__ == "__main__":
    main()
