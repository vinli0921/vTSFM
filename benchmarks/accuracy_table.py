"""Summarize accuracy-profile runs as model x signal x dtype tables.

Pass one or more ``scripts/test-models.sh --profile accuracy`` output
directories (e.g. an FP32 run and a BF16 run). Each
``<run>/<adapter>/accuracy.json`` contributes per-window native, vTSFM,
seasonal-naive and mean-of-history scores; ``run.json`` supplies the dtype.
Rows group windows by signal: the ``<signal>/origin-<slot>`` item-ID prefix,
or the case name.

Reported scores are the official native model's. vTSFM must match native
within the parity gate, so the last column only confirms that; signals whose
cases failed the gate are marked there (``parity_failed_cases`` in the CSV).
SQL is the scaled quantile loss over all nine quantiles, the harness's quantile
approximation of a scaled CRPS.

Aggregates. ``MASE``/``SQL`` keep the original equal-weight mean of per-window
ratios, which a window with a tiny seasonal scale can dominate. Two aggregates
resist that: the ratio of means (summed absolute error over summed scale, i.e.
a scale-weighted mean) and the geometric mean of per-window MASE. Undefined
scaled scores (zero or too-short seasonal scale) stay out of every scaled
aggregate and are counted instead; a zero per-window error is counted and
excluded from the geometric mean. The comparison section ranks models and both
baselines within each window by median MAE (all competitors share the window's
scale) and gives fixed-seed paired-bootstrap 95% intervals: each resample draws
windows with replacement and scores every competitor on the same draw. Windows
from one asset are correlated, so the intervals are approximate.

Flags. A window is ``off_state`` when at least half of its observed future
labels are exactly zero (an asset that is switched off); aggregates are also
reported without those windows. Signals in ``UNFORECASTABLE`` are scored and
shown but never named in ranking claims. Flags need each run's ``cases.json``.

"Filled history" is the mean share of history slots that were causally
forward-filled, read from ``cases.json`` provenance. By default the seasonal
scale uses every history pair, as scored. ``--scale observed`` re-expresses MASE
and SQL over pairs whose two values were both recorded, so forward-filled runs
cannot shrink the scale. This is exact (both metrics divide by the scale) and
needs no rerun. Runs made before the mean-of-history baseline existed get it
recomputed from ``cases.json`` (this one path needs NumPy); everything else is
standard library only.

    python -m benchmarks.accuracy_table RUN_DIR [RUN_DIR ...] [--scale observed] [--output DIR]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import re

from benchmarks.model_evaluation import SCALES, history_observed, seasonal_scale


MODELS = ("timesfm25", "toto2", "moirai2")
BASELINES = ("seasonal_naive", "mean_history")
DTYPE_NAMES = {"float32": "fp32", "fp32": "fp32", "tf32": "tf32", "bf16": "bf16", "bfloat16": "bf16"}
DTYPE_ORDER = {"fp32": 0, "tf32": 1, "bf16": 2}
# A window is off-state when at least this share of observed future labels is exactly zero.
OFF_STATE_ZERO_SHARE = 0.5
# Signals shown but excluded from ranking claims, with the reason printed in the table.
UNFORECASTABLE = {
    "chiller_6/supply_temperature": "noise-like (lag-1 autocorrelation -0.01 to 0.12); "
                                    "the history mean scores within the models' range",
}
RESAMPLES = 2000
SEED = 0
# Markdown labels; the CSV keeps the plain subset names.
SUBSET_LABELS = {"all": "all", "excluding_off_state": "excluding off-state (post hoc)"}
# Below this many windows a bootstrap interval is too coarse to support any claim.
MIN_CLAIM_WINDOWS = 5
COLUMNS = ["model", "dtype", "signal", "scale", "series", "scaled_series", "zero_scale_series",
           "filled_history_pct", "MAE", "MASE", "SQL", "naive_MASE", "naive_SQL",
           "max_abs_vtsfm_minus_native_SQL", "parity_failed_cases",
           "off_state_windows", "unforecastable", "MASE_ratio_of_means", "SQL_ratio_of_means",
           "MASE_geometric_mean", "geometric_mean_windows", "naive_MASE_ratio_of_means",
           "mean_history_MASE", "mean_history_MASE_ratio_of_means"]
SOURCES = {"native": "native", "vtsfm": "vtsfm", "seasonal_naive": "seasonal_naive", "mean_history": "mean_history"}
METRICS = ("MAE", "MASE", "SQL")
WINDOW_COLUMNS = (["model", "dtype", "signal", "case", "item_id", "origin", "scale", "seasonal_scale", "scale_status",
                   "observed_targets", "future_zero_share", "off_state", "unforecastable", "filled_history_pct",
                   "parity"]
                  + [f"{source}_{metric}" for source in SOURCES for metric in METRICS])
COMPARISON_COLUMNS = ["dtype", "signal", "subset", "competitor", "windows", "scaled_windows",
                      "MASE_ratio_of_means", "MASE_ratio_of_means_low", "MASE_ratio_of_means_high",
                      "MASE_geometric_mean", "MASE_geometric_mean_low", "MASE_geometric_mean_high",
                      "geometric_mean_windows", "mean_rank", "probability_lowest", "lowest",
                      "ratio_to_lowest", "ratio_to_lowest_low", "ratio_to_lowest_high", "claim"]


def _signal(case_name: str, item_id: str) -> str:
    if "/origin-" in item_id:
        return item_id.rsplit("/origin-", 1)[0]
    return re.sub(r"-window-\d+$", "", case_name)


def _origin(source, row, item_id: str):
    """The window's recorded forecast origin: per-row for signal cases, shared for the
    MetroPT fixture, per-series for fev; else parsed from ``.../origin-<slot>[/...]``."""
    provenance = (source or {}).get("provenance") or {}
    if row is not None:
        if isinstance(provenance.get("origins"), list):
            return provenance["origins"][row]
        if isinstance(provenance.get("series"), list) and row < len(provenance["series"]):
            if provenance["series"][row].get("origin") is not None:
                return provenance["series"][row]["origin"]
    if provenance.get("origin") is not None:
        return provenance["origin"]
    match = re.search(r"/origin-(\d+)(?:/|$)", item_id)
    return int(match.group(1)) if match else None


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


def _future_zero_share(source, row):
    """Share of observed future labels that are exactly zero, or None without the case."""
    if source is None or row is None:
        return None
    labels = [value for value, flag in zip(source["evaluation"]["targets"][row], source["evaluation"]["observed"][row])
              if flag]
    return sum(value == 0 for value in labels) / len(labels) if labels else None


def _mean_history_rows(source, scale):
    """Score the mean-of-history baseline for an older run that did not record it."""
    from benchmarks.model_evaluation import score_mean_history

    return score_mean_history(source, scale=scale)["per_series"]


def collect_windows(run_dirs, scale="all") -> list[dict]:
    """Return one row per model, dtype and window across all given runs.

    Each row keeps the window's scores for every source (``native``, ``vtsfm``,
    ``seasonal_naive``, ``mean_history``) under those keys, plus flags.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    windows = []
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
                if "mean_history" in case:
                    history = case["mean_history"]["per_series"]
                    history_scale = "all"  # Recorded runs score with the default scale.
                elif source is not None:
                    history, history_scale = _mean_history_rows(source, scale), scale
                else:
                    history, history_scale = [None] * len(case["native"]["per_series"]), scale
                rows = zip(case["native"]["per_series"], case["vtsfm"]["per_series"],
                           case["seasonal_naive"]["per_series"], history, strict=True)
                for native, served, naive, mean in rows:
                    if not native["item_id"] == served["item_id"] == naive["item_id"]:
                        raise ValueError(f"{folder}: per-series rows are not aligned in {case['case']}")
                    if mean is not None and mean["item_id"] != native["item_id"]:
                        raise ValueError(f"{folder}: mean-of-history rows are not aligned in {case['case']}")
                    row = index.get(native["item_id"])
                    mask = masks[row] if masks is not None and row is not None else None
                    if scale == "observed":
                        if mask is None:
                            raise ValueError(f"{folder}: {case['case']} does not record history imputation")
                        recorded, _ = seasonal_scale(source["data"]["inputs"][row]["target"],
                                                     source["evaluation"]["seasonality"], mask)
                        native, served, naive = (_rescale(entry, recorded) for entry in (native, served, naive))
                        if mean is not None and history_scale != scale:
                            mean = _rescale(mean, recorded)
                    signal = _signal(case["case"], native["item_id"])
                    zero_share = _future_zero_share(source, row)
                    windows.append({
                        "model": folder.name, "dtype": dtype, "signal": signal, "case": case["case"],
                        "item_id": native["item_id"], "origin": _origin(source, row, native["item_id"]), "scale": scale,
                        "seasonal_scale": native.get("seasonal_scale"), "scale_status": native.get("scale_status"),
                        "observed_targets": native.get("observed_targets"), "future_zero_share": zero_share,
                        "off_state": None if zero_share is None else zero_share >= OFF_STATE_ZERO_SHARE,
                        "unforecastable": signal in UNFORECASTABLE,
                        "filled_history": None if mask is None else mask.count(False) / len(mask),
                        "parity": case.get("parity", "passed"),
                        "native": native, "vtsfm": served, "seasonal_naive": naive, "mean_history": mean,
                    })
    return windows


def _ratio_of_means(rows, numerator="MAE"):
    """Summed error over summed scale across windows with a valid scale (MASE) or SQL x scale."""
    pairs = [(row[numerator] if numerator == "MAE" else row["SQL"] * row["seasonal_scale"], row["seasonal_scale"])
             for row in rows if row is not None and row.get("MASE") is not None and row.get("seasonal_scale")
             and (numerator == "MAE" or row.get("SQL") is not None)]
    total = sum(scale for _, scale in pairs)
    return sum(error for error, _ in pairs) / total if pairs and total > 0 else None


def _geometric_mean(values):
    values = [value for value in values if value is not None and value > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def collect(run_dirs, scale="all", windows=None) -> list[dict]:
    """Return one summary row per model, dtype and signal across all given runs."""
    windows = collect_windows(run_dirs, scale) if windows is None else windows
    groups = {}
    for window in windows:
        groups.setdefault((window["model"], window["dtype"], window["signal"]), []).append(window)
    table = []
    for (model, dtype, signal), group in groups.items():
        native = [window["native"] for window in group]
        served = [window["vtsfm"] for window in group]
        naive = [window["seasonal_naive"] for window in group]
        history = [window["mean_history"] for window in group]
        differences = [abs(a["SQL"] - b["SQL"]) for a, b in zip(served, native)
                       if a["SQL"] is not None and b["SQL"] is not None]
        filled = _mean(window["filled_history"] for window in group)
        known_history = [row for row in history if row is not None]
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
            "parity_failed_cases": len({window["case"] for window in group if window["parity"] != "passed"}),
            "off_state_windows": sum(window["off_state"] is True for window in group),
            "unforecastable": signal in UNFORECASTABLE,
            "MASE_ratio_of_means": _ratio_of_means(native),
            "SQL_ratio_of_means": _ratio_of_means(native, "SQL"),
            "MASE_geometric_mean": _geometric_mean(row["MASE"] for row in native),
            "geometric_mean_windows": sum(row["MASE"] is not None and row["MASE"] > 0 for row in native),
            "naive_MASE_ratio_of_means": _ratio_of_means(naive),
            "mean_history_MASE": _mean(row["MASE"] for row in known_history) if known_history else None,
            "mean_history_MASE_ratio_of_means": _ratio_of_means(known_history) if known_history else None,
        })
    order = {model: index for index, model in enumerate(MODELS)}
    return sorted(table, key=lambda row: (row["signal"], order.get(row["model"], len(MODELS)), row["model"],
                                          DTYPE_ORDER.get(row["dtype"], len(DTYPE_ORDER)), row["dtype"]))


def _percentile(values, fraction):
    """Linear-interpolation percentile of a nonempty list."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _ranks(errors):
    """Average ranks (1 = lowest error) for one window's competitor errors."""
    ordered = sorted(range(len(errors)), key=lambda index: errors[index])
    ranks = [0.0] * len(errors)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and errors[ordered[end + 1]] == errors[ordered[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[ordered[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _statistics(names, errors, scales, draw):
    """Ratio-of-means and geometric-mean MASE for each competitor on one draw of windows."""
    scaled = [window for window in draw if scales[window]]
    total = sum(scales[window] for window in scaled)
    ratio = {name: (sum(errors[name][window] for window in scaled) / total if scaled and total > 0 else None)
             for name in names}
    # A zero error makes a log undefined; drop that window for every competitor (paired).
    usable = [window for window in scaled if all(errors[name][window] > 0 for name in names)]
    geometric = {name: (math.exp(sum(math.log(errors[name][window] / scales[window]) for window in usable)
                                 / len(usable)) if usable else None) for name in names}
    return ratio, geometric, len(usable)


def _ratio_to_lowest(value, lowest):
    """Competitor score over the lowest score; undefined when the lowest is zero,
    except that a competitor also at zero is tied with it."""
    if value is None or lowest is None:
        return None
    if lowest == 0:
        return 1.0 if value == 0 else None
    return value / lowest


def compare(windows, *, resamples=RESAMPLES, seed=SEED) -> list[dict]:
    """Paired comparison of native models and both baselines per dtype and signal.

    Competitors are every model with native scores for the signal, plus the
    seasonal-naive and mean-of-history baselines (computed from identical cases,
    taken from the first model). Only windows every competitor scored are used.
    Results are reported for all windows and, when any window is off-state,
    without off-state windows. Unforecastable signals get no ranking claim.
    """
    groups = {}
    for window in windows:
        # A series can recur across cases (fev cutoffs), so the case is part of the pairing key.
        key = (window["case"], window["item_id"])
        models = groups.setdefault((window["dtype"], window["signal"]), {}).setdefault(window["model"], {})
        if key in models:
            raise ValueError(f"Duplicate window {key} for {window['model']} {window['dtype']}")
        models[key] = window
    order = {model: index for index, model in enumerate(MODELS)}
    results = []
    for (dtype, signal), by_model in sorted(groups.items(), key=lambda item: (
            item[0][1], DTYPE_ORDER.get(item[0][0], len(DTYPE_ORDER)), item[0][0])):
        models = sorted(by_model, key=lambda model: (order.get(model, len(MODELS)), model))
        first = by_model[models[0]]
        items = [item for item in first if all(item in by_model[model] for model in models)
                 and all(by_model[model][item]["native"]["MAE"] is not None for model in models)
                 and first[item]["seasonal_naive"]["MAE"] is not None
                 and first[item]["mean_history"] is not None and first[item]["mean_history"]["MAE"] is not None]
        items.sort(key=lambda item: (first[item]["origin"] is None, first[item]["origin"] or 0, item))
        subsets = [("all", items)]
        on_state = [item for item in items if first[item]["off_state"] is not True]
        if on_state and len(on_state) < len(items):
            subsets.append(("excluding_off_state", on_state))
        for subset, chosen in subsets:
            if not chosen:
                continue
            names = list(models) + list(BASELINES)
            errors = {model: [by_model[model][item]["native"]["MAE"] for item in chosen] for model in models}
            for baseline in BASELINES:
                errors[baseline] = [first[item][baseline]["MAE"] for item in chosen]
            scales = [first[item]["seasonal_scale"] if first[item]["native"]["MASE"] is not None else None
                      for item in chosen]
            everything = list(range(len(chosen)))
            # Ratio-of-means only uses windows with a positive seasonal scale.
            scored_windows = sum(1 for scale in scales if scale)
            ratio, geometric, geometric_windows = _statistics(names, errors, scales, everything)
            ranks = [_ranks([errors[name][window] for name in names]) for window in everything]
            mean_rank = {name: sum(rank[position] for rank in ranks) / len(ranks) for position, name in enumerate(names)}
            # One generator per signal/dtype/subset keeps results independent of input order.
            generator = random.Random(f"{seed}:{dtype}:{signal}:{subset}")
            samples = {name: {"ratio": [], "geometric": []} for name in names}
            lowest_counts = dict.fromkeys(names, 0.0)
            decided = 0  # Draws with a defined winner.
            ratio_samples = {name: [] for name in names}
            scored = [name for name in names if ratio[name] is not None]
            lowest = min(scored, key=lambda name: ratio[name]) if scored else None
            for _ in range(resamples):
                draw = [generator.randrange(len(chosen)) for _ in chosen]
                draw_ratio, draw_geometric, _ = _statistics(names, errors, scales, draw)
                for name in names:
                    if draw_ratio[name] is not None:
                        samples[name]["ratio"].append(draw_ratio[name])
                    if draw_geometric[name] is not None:
                        samples[name]["geometric"].append(draw_geometric[name])
                defined = [name for name in names if draw_ratio[name] is not None]
                if defined:
                    best = min(draw_ratio[name] for name in defined)
                    winners = [name for name in defined if draw_ratio[name] == best]
                    for name in winners:
                        lowest_counts[name] += 1 / len(winners)
                    decided += 1
                if lowest is not None and draw_ratio[lowest]:
                    for name in names:
                        if draw_ratio[name] is not None:
                            ratio_samples[name].append(draw_ratio[name] / draw_ratio[lowest])

            def interval(values):
                return (_percentile(values, 0.025), _percentile(values, 0.975)) if values else (None, None)

            group_rows = []
            for name in names:
                ratio_low, ratio_high = interval(samples[name]["ratio"])
                geometric_low, geometric_high = interval(samples[name]["geometric"])
                relative = _ratio_to_lowest(ratio[name], None if lowest is None else ratio[lowest])
                relative_low, relative_high = interval(ratio_samples[name]) if name != lowest else (1.0, 1.0)
                if signal in UNFORECASTABLE:
                    claim = "no claim: unforecastable signal"
                elif scored_windows < MIN_CLAIM_WINDOWS:
                    claim = f"no claim: fewer than {MIN_CLAIM_WINDOWS} scored windows"
                elif name == lowest:
                    claim = "lowest"
                elif relative_low is None:
                    claim = "undefined"
                else:
                    # Only an interval wholly above 1 separates this competitor from the lowest.
                    claim = "higher than lowest" if relative_low > 1 else "not distinguishable from lowest"
                group_rows.append({
                    "dtype": dtype, "signal": signal, "subset": subset, "competitor": name,
                    "windows": len(chosen), "scaled_windows": sum(scale is not None for scale in scales),
                    "MASE_ratio_of_means": ratio[name], "MASE_ratio_of_means_low": ratio_low,
                    "MASE_ratio_of_means_high": ratio_high, "MASE_geometric_mean": geometric[name],
                    "MASE_geometric_mean_low": geometric_low, "MASE_geometric_mean_high": geometric_high,
                    "geometric_mean_windows": geometric_windows, "mean_rank": mean_rank[name],
                    "probability_lowest": lowest_counts[name] / decided if decided else None,
                    "lowest": lowest, "ratio_to_lowest": relative,
                    "ratio_to_lowest_low": relative_low, "ratio_to_lowest_high": relative_high, "claim": claim,
                })
            ties = [row["competitor"] for row in group_rows if row["claim"] == "not distinguishable from lowest"]
            undefined = [row["competitor"] for row in group_rows if row["claim"] == "undefined"]
            for row in group_rows:
                if row["claim"] == "lowest":
                    parts = []
                    if ties:
                        parts.append("not distinguishable from " + ", ".join(ties))
                    if undefined:
                        parts.append("undefined against " + ", ".join(undefined))
                    row["claim"] = "lowest; " + ("; ".join(parts) if parts else "every other competitor higher")
            results.extend(group_rows)
    return results


def _number(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.2e}" if 0 < abs(value) < 10 ** -digits else f"{value:.{digits}f}"


def _interval(value, low, high, digits=3):
    if value is None:
        return "n/a"
    if low is None:
        return _number(value, digits)
    return f"{_number(value, digits)} [{_number(low, digits)}, {_number(high, digits)}]"


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


def markdown(table, comparison=None, *, resamples=RESAMPLES, seed=SEED) -> str:
    lines = []
    if any(row["scale"] == "observed" for row in table):
        lines += ["MASE and SQL use seasonal scales over recorded history pairs only (`--scale observed`).", ""]
    lines += ["| Signal | Model | dtype | Windows (scaled) | Filled history | MAE | MASE | SQL | MASE ratio of means | "
              "MASE geo-mean | Naive MASE | Naive SQL | History-mean MASE | max \\|vTSFM−native\\| SQL |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        counts = f"{row['series']} ({row['scaled_series']})"
        if row["zero_scale_series"]:
            counts += f", {row['zero_scale_series']} zero-scale"
        if row.get("off_state_windows"):
            counts += f", {row['off_state_windows']} off-state"
        signal = row["signal"] + (" (unforecastable)" if row.get("unforecastable") else "")
        filled = "n/a" if row["filled_history_pct"] is None else f"{row['filled_history_pct']:.1f}%"
        lines.append(f"| {signal} | {row['model']} | {row['dtype']} | {counts} | {filled} | {_number(row['MAE'])} | "
                     f"{_number(row['MASE'])} | {_number(row['SQL'])} | {_number(row.get('MASE_ratio_of_means'))} | "
                     f"{_number(row.get('MASE_geometric_mean'))} | {_number(row['naive_MASE'])} | "
                     f"{_number(row['naive_SQL'])} | {_number(row.get('mean_history_MASE'))} | "
                     f"{_number(row['max_abs_vtsfm_minus_native_SQL'], 6)}"
                     f"{' (parity failed)' if row['parity_failed_cases'] else ''} |")
    flagged = [signal for signal in UNFORECASTABLE if any(row["signal"] == signal for row in table)]
    if flagged:
        lines += [""] + [f"`{signal}` is unforecastable on this data: {UNFORECASTABLE[signal]}. "
                         "It is scored but excluded from ranking claims." for signal in flagged]
    if comparison:
        lines += ["", f"**Paired comparison** (MASE; 95% percentile intervals from {resamples} paired bootstrap "
                      f"resamples of windows, seed {seed}). Windows are resampled independently, but their "
                      "histories overlap heavily, so the intervals are approximate and probably too narrow. "
                      "Off-state is defined from the future labels, so the subset excluding it is a post hoc "
                      "diagnostic. Rank 1 is the lowest median MAE within a window.", "",
                  "| Signal | dtype | Subset | Windows | Competitor | Ratio of means | Geo-mean | Mean rank | "
                  "P(lowest) | ÷ lowest | Claim |",
                  "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in comparison:
            lines.append(f"| {row['signal']} | {row['dtype']} | {SUBSET_LABELS.get(row['subset'], row['subset'])} | "
                         f"{row['windows']} | "
                         f"{row['competitor']} | "
                         f"{_interval(row['MASE_ratio_of_means'], row['MASE_ratio_of_means_low'], row['MASE_ratio_of_means_high'])} | "
                         f"{_interval(row['MASE_geometric_mean'], row['MASE_geometric_mean_low'], row['MASE_geometric_mean_high'])} | "
                         f"{_number(row['mean_rank'], 2)} | {_number(row['probability_lowest'], 2)} | "
                         f"{_interval(row['ratio_to_lowest'], row['ratio_to_lowest_low'], row['ratio_to_lowest_high'], 2)} | "
                         f"{row['claim']} |")
    spread = model_spread(table)
    if spread:
        lines += ["", "**Model spread** (largest ÷ smallest mean of per-window ratios on the sampled windows; "
                      "see the paired comparison before reading it as a ranking)", "",
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


def window_rows(windows) -> list[dict]:
    """Flatten per-window results for ``*_windows.csv``."""
    rows = []
    for window in windows:
        row = {key: window[key] for key in WINDOW_COLUMNS if key in window}
        row["filled_history_pct"] = None if window["filled_history"] is None else 100 * window["filled_history"]
        for source in SOURCES:
            scores = window[source] or {}
            for metric in METRICS:
                row[f"{source}_{metric}"] = scores.get(metric)
        rows.append(row)
    return rows


def _csv(rows, columns):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="test-models.sh accuracy output directories")
    parser.add_argument("--scale", choices=SCALES, default="all",
                        help="Seasonal scale for MASE/SQL: every history pair (default) or recorded pairs only")
    parser.add_argument("--resamples", type=int, default=RESAMPLES, help="Paired bootstrap resamples (0 disables)")
    parser.add_argument("--seed", type=int, default=SEED, help="Bootstrap seed")
    parser.add_argument("--output", type=Path,
                        help="Also write accuracy_table.md/.csv plus _windows.csv and _comparison.csv here "
                             "(accuracy_table_observed_scale* with --scale observed)")
    args = parser.parse_args(argv)
    if args.resamples < 0:
        parser.error("--resamples must be nonnegative")
    windows = collect_windows(args.runs, args.scale)
    table = collect(args.runs, args.scale, windows)
    comparison = compare(windows, resamples=args.resamples, seed=args.seed)
    text = markdown(table, comparison, resamples=args.resamples, seed=args.seed)
    print(text, end="")
    if args.output is not None:
        stem = "accuracy_table" if args.scale == "all" else "accuracy_table_observed_scale"
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{stem}.md").write_text(text)
        (args.output / f"{stem}.csv").write_text(_csv(table, COLUMNS))
        (args.output / f"{stem}_windows.csv").write_text(_csv(window_rows(windows), WINDOW_COLUMNS))
        (args.output / f"{stem}_comparison.csv").write_text(_csv(comparison, COMPARISON_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
