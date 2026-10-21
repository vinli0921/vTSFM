#!/usr/bin/env python3
"""Prepare pinned AssetOpsBench numeric fixtures without generating observations.

Standard library only. Writes a NumPy-compatible data.npz containing
``series`` with shape [7, 5116], dtype float32, and a provenance manifest.
The source has 560 wholly empty hourly rows. They are causally forward-filled
from the last available observation, with all imputed indices documented.
The channels share one asset and time axis; overlapping windows are correlated.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import urllib.request
import zipfile


COMMIT = "e844f6468c6c68b328022e7ca8e31da140842cc7"
SOURCE_PATH = "src/couchdb/scenarios_data/shared/iot/metro_pump_1.json"
SOURCE_URL = f"https://raw.githubusercontent.com/IBM/AssetOpsBench/{COMMIT}/{SOURCE_PATH}"
SOURCE_SHA256 = "363d6daf873c614fcb4a798d4e86aa648d60ae2b6c735ce5d8388b0d4b3c2fdf"
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


def fetch_source(source_file: Path | None, cache_path: Path) -> bytes:
    if source_file is not None:
        content = source_file.read_bytes()
    elif cache_path.exists():
        content = cache_path.read_bytes()
    else:
        request = urllib.request.Request(
            SOURCE_URL, headers={"User-Agent": "AssetOpsBench-numeric-benchmark/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    actual = sha256(content)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Source checksum mismatch: expected {SOURCE_SHA256}, got {actual}")
    return content


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=Path("data"))
    parser.add_argument("--source-file", type=Path, help="Use an offline fixture copy; the pinned SHA256 is still verified")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = prepare(output, args.source_file)
    print(json.dumps({"output_dir": str(output.resolve()), "array": manifest["array"], "benchmark": manifest["benchmark"]}, indent=2))


if __name__ == "__main__":
    main()
