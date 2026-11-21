"""Summarize accuracy-profile runs as a model x signal x dtype table.

Standard library only. Pass one or more ``scripts/test-models.sh --profile
accuracy`` output directories (e.g. an FP32 run and a BF16 run). Each
``<run>/<adapter>/accuracy.json`` contributes per-series native, vTSFM and
seasonal-naive scores; ``run.json`` supplies the dtype. Rows group series by
signal: the ``<signal>/origin-<slot>`` item-ID prefix, or the case name.

Reported scores are the official native model's. vTSFM must match native
within the parity gate, so the last column only confirms that; signals whose
cases failed the gate are marked there (``parity_failed_cases`` in the CSV).
Means are equal weight over series with a defined value; undefined scaled
scores (zero or too-short seasonal scale) stay out of the mean and are counted
instead. SQL is the scaled quantile loss over all nine quantiles, the
harness's quantile approximation of a scaled CRPS.

"Filled history" is the mean share of history slots that were causally
forward-filled, read from each run's ``cases.json`` provenance. By default the
seasonal scale uses every history pair, as scored. ``--scale observed``
re-expresses MASE and SQL over pairs whose two values were both recorded, so
forward-filled runs cannot shrink the scale. This is exact (both metrics divide
by the scale) and needs no rerun.

    python -m benchmarks.accuracy_table RUN_DIR [RUN_DIR ...] [--scale observed] [--output DIR]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re

from benchmarks.model_evaluation import SCALES, history_observed, seasonal_scale


MODELS = ("timesfm25", "toto2", "moirai2")
DTYPE_NAMES = {"float32": "fp32", "fp32": "fp32", "tf32": "tf32", "bf16": "bf16", "bfloat16": "bf16"}
DTYPE_ORDER = {"fp32": 0, "tf32": 1, "bf16": 2}
COLUMNS = ["model", "dtype", "signal", "scale", "series", "scaled_series", "zero_scale_series",
           "filled_history_pct", "MAE", "MASE", "SQL", "naive_MASE", "naive_SQL",
           "max_abs_vtsfm_minus_native_SQL", "parity_failed_cases"]


def _signal(case_name: str, item_id: str) -> str:
    if "/origin-" in item_id:
        return item_id.rsplit("/origin-", 1)[0]
    return re.sub(r"-window-\d+$", "", case_name)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _rescale(row, scale):
    """Re-express one series' MASE and SQL over another seasonal scale."""
    old, usable = row.get("seasonal_scale"), scale is not None and scale > 0
    return {**row, "seasonal_scale": scale,
            "scale_status": "valid" if usable else "zero" if scale == 0 else "insufficient_history",
            "MASE": row["MAE"] / scale if usable and row["MAE"] is not None else None,
            "SQL": row["SQL"] * old / scale if usable and row["SQL"] is not None and old else None}


def collect(run_dirs, scale="all") -> list[dict]:
    """Return one row per model, dtype and signal across all given runs."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    groups = {}
    for run_dir in map(Path, run_dirs):
        adapters = [path.parent for path in sorted(run_dir.glob("*/accuracy.json"))]
        if not adapters:
            raise ValueError(f"No <adapter>/accuracy.json under {run_dir}")
        for folder in adapters:
            run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {}
            dtype = DTYPE_NAMES.get(str(run.get("precision", "float32")), str(run.get("precision")))
            # The exact cases this run scored, with their imputation provenance.
            cases_file = folder / "cases.json"
            sources = ({entry["name"]: entry for entry in json.loads(cases_file.read_text())["cases"]}
                       if cases_file.exists() else {})
            if scale == "observed" and not sources:
                raise ValueError(f"--scale observed needs {cases_file}")
            for case in json.loads((folder / "accuracy.json").read_text())["cases"]:
                source = sources.get(case["case"])
                masks = history_observed(source) if source else None
                index = {item["item_id"]: row for row, item in enumerate(source["data"]["inputs"])} if source else {}
                rows = zip(case["native"]["per_series"], case["vtsfm"]["per_series"],
                           case["seasonal_naive"]["per_series"], strict=True)
                for native, served, naive in rows:
                    if not native["item_id"] == served["item_id"] == naive["item_id"]:
                        raise ValueError(f"{folder}: per-series rows are not aligned in {case['case']}")
                    row = index.get(native["item_id"])
                    mask = masks[row] if masks is not None and row is not None else None
                    if scale == "observed":
                        if mask is None:
                            raise ValueError(f"{folder}: {case['case']} does not record history imputation")
                        recorded, _ = seasonal_scale(source["data"]["inputs"][row]["target"],
                                                     source["evaluation"]["seasonality"], mask)
                        native, served, naive = (_rescale(entry, recorded) for entry in (native, served, naive))
                    key = (folder.name, dtype, _signal(case["case"], native["item_id"]))
                    group = groups.setdefault(key, {"native": [], "vtsfm": [], "naive": [], "failed": set(),
                                                    "filled": []})
                    group["filled"].append(None if mask is None else mask.count(False) / len(mask))
                    if case.get("parity", "passed") != "passed":
                        group["failed"].add(case["case"])
                    group["native"].append(native)
                    group["vtsfm"].append(served)
                    group["naive"].append(naive)
    table = []
    for (model, dtype, signal), group in groups.items():
        native, served, naive = group["native"], group["vtsfm"], group["naive"]
        differences = [abs(a["SQL"] - b["SQL"]) for a, b in zip(served, native)
                       if a["SQL"] is not None and b["SQL"] is not None]
        filled = _mean(group["filled"])
        table.append({
            "model": model, "dtype": dtype, "signal": signal, "scale": scale, "series": len(native),
            "scaled_series": sum(row["MASE"] is not None for row in native),
            "zero_scale_series": sum(row.get("scale_status") == "zero" for row in native),
            "filled_history_pct": None if filled is None else 100 * filled,
            "MAE": _mean(row["MAE"] for row in native),
            "MASE": _mean(row["MASE"] for row in native),
            "SQL": _mean(row["SQL"] for row in native),
            "naive_MASE": _mean(row["MASE"] for row in naive),
            "naive_SQL": _mean(row["SQL"] for row in naive),
            "max_abs_vtsfm_minus_native_SQL": max(differences) if differences else None,
            "parity_failed_cases": len(group["failed"]),
        })
    order = {model: index for index, model in enumerate(MODELS)}
    return sorted(table, key=lambda row: (row["signal"], order.get(row["model"], len(MODELS)), row["model"],
                                          DTYPE_ORDER.get(row["dtype"], len(DTYPE_ORDER)), row["dtype"]))


def _number(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.2e}" if 0 < abs(value) < 10 ** -digits else f"{value:.{digits}f}"


def precision_deltas(table) -> list[dict]:
    """BF16/TF32 minus FP32 scores where both runs exist for a model and signal."""
    baseline = {(row["model"], row["signal"]): row for row in table if row["dtype"] == "fp32"}
    deltas = []
    for row in table:
        reference = baseline.get((row["model"], row["signal"]))
        if row["dtype"] == "fp32" or reference is None:
            continue
        delta = {"model": row["model"], "signal": row["signal"], "dtype": row["dtype"]}
        for metric in ("MASE", "SQL"):
            new, old = row[metric], reference[metric]
            delta[f"delta_{metric}"] = None if new is None or old is None else new - old
            delta[f"relative_{metric}"] = None if new is None or not old else (new - old) / old
        deltas.append(delta)
    return deltas


def model_spread(table) -> list[dict]:
    """Smallest and largest sampled mean per signal and dtype."""
    spread = {}
    for row in table:
        if row["MASE"] is not None:
            spread.setdefault((row["signal"], row["dtype"]), []).append(row)
    result = []
    for (signal, dtype), rows in sorted(spread.items(), key=lambda item: (
            item[0][0], DTYPE_ORDER.get(item[0][1], len(DTYPE_ORDER)), item[0][1])):
        if len(rows) < 2:
            continue
        entry = {"signal": signal, "dtype": dtype}
        for metric in ("MASE", "SQL"):
            scored = sorted((row for row in rows if row[metric] is not None), key=lambda row: row[metric])
            best, worst = scored[0], scored[-1]
            entry[f"best_{metric}"] = f"{best['model']} {_number(best[metric])}"
            entry[f"worst_{metric}"] = f"{worst['model']} {_number(worst[metric])}"
            entry[f"worst_over_best_{metric}"] = worst[metric] / best[metric] if best[metric] else None
        result.append(entry)
    return result


def markdown(table) -> str:
    lines = []
    if any(row["scale"] == "observed" for row in table):
        lines += ["MASE and SQL use seasonal scales over recorded history pairs only (`--scale observed`).", ""]
    lines += ["| Signal | Model | dtype | Series (scaled) | Filled history | MAE | MASE | SQL | Naive MASE | Naive SQL | max \\|vTSFM−native\\| SQL |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        counts = f"{row['series']} ({row['scaled_series']})"
        if row["zero_scale_series"]:
            counts += f", {row['zero_scale_series']} zero-scale"
        filled = "n/a" if row["filled_history_pct"] is None else f"{row['filled_history_pct']:.1f}%"
        lines.append(f"| {row['signal']} | {row['model']} | {row['dtype']} | {counts} | {filled} | {_number(row['MAE'])} | "
                     f"{_number(row['MASE'])} | {_number(row['SQL'])} | {_number(row['naive_MASE'])} | "
                     f"{_number(row['naive_SQL'])} | {_number(row['max_abs_vtsfm_minus_native_SQL'], 6)}"
                     f"{' (parity failed)' if row['parity_failed_cases'] else ''} |")
    spread = model_spread(table)
    if spread:
        lines += ["", "**Model spread** (largest ÷ smallest mean on the sampled windows)", "",
                  "| Signal | dtype | Lowest MASE | Highest MASE | Ratio | Lowest SQL | Highest SQL | Ratio |",
                  "| --- | --- | --- | --- | ---: | --- | --- | ---: |"]
        for entry in spread:
            lines.append(f"| {entry['signal']} | {entry['dtype']} | {entry['best_MASE']} | {entry['worst_MASE']} | "
                         f"{_number(entry['worst_over_best_MASE'], 2)} | {entry['best_SQL']} | {entry['worst_SQL']} | "
                         f"{_number(entry['worst_over_best_SQL'], 2)} |")
    deltas = precision_deltas(table)
    if deltas:
        lines += ["", "**Reduced precision minus FP32** (positive means worse)", "",
                  "| Signal | Model | dtype | ΔMASE | ΔMASE % | ΔSQL | ΔSQL % |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for delta in deltas:
            percent = {metric: "n/a" if delta[f"relative_{metric}"] is None else f"{100 * delta[f'relative_{metric}']:+.2f}%"
                       for metric in ("MASE", "SQL")}
            lines.append(f"| {delta['signal']} | {delta['model']} | {delta['dtype']} | {_number(delta['delta_MASE'], 4)} | "
                         f"{percent['MASE']} | {_number(delta['delta_SQL'], 4)} | {percent['SQL']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="test-models.sh accuracy output directories")
    parser.add_argument("--scale", choices=SCALES, default="all",
                        help="Seasonal scale for MASE/SQL: every history pair (default) or recorded pairs only")
    parser.add_argument("--output", type=Path,
                        help="Also write accuracy_table.md/.csv here (accuracy_table_observed_scale.* with --scale observed)")
    args = parser.parse_args(argv)
    table = collect(args.runs, args.scale)
    text = markdown(table)
    print(text, end="")
    if args.output is not None:
        stem = "accuracy_table" if args.scale == "all" else "accuracy_table_observed_scale"
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{stem}.md").write_text(text)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
        (args.output / f"{stem}.csv").write_text(buffer.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
