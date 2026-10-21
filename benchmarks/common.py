"""Shared numerical workload and statistical definitions (CPU-only on import)."""
import hashlib
import json
import math
from pathlib import Path
import statistics

QUANTILES = [0.1, 0.5, 0.9]
HORIZON = 96


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(latencies, batch_size):
    if not latencies or any(not math.isfinite(value) or value <= 0 for value in latencies):
        raise ValueError("Request latencies must be finite and positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mean = statistics.mean(latencies)
    return {
        "n": len(latencies), "mean_ms": mean * 1000,
        "p50_ms": percentile(latencies, 0.5) * 1000,
        "p95_ms": percentile(latencies, 0.95) * 1000,
        "request_per_s": 1 / mean, "series_per_s": batch_size / mean,
    }


def prediction_array(response, batch_size, horizon=HORIZON):
    predictions = response["data"]["predictions"]
    if len(predictions) != batch_size:
        raise AssertionError((len(predictions), batch_size))
    result = []
    for prediction in predictions:
        for key in ["mean", "0.1", "0.5", "0.9"]:
            values = prediction[key]
            if len(values) != horizon:
                raise AssertionError((key, len(values), horizon))
            if not all(math.isfinite(float(value)) for value in values):
                raise AssertionError(f"Nonfinite prediction values in {key}")
            result.extend(values)
    return result


def quantile_crossings(response):
    return sum(low > middle + 1e-6 or middle > high + 1e-6
               for prediction in response["data"]["predictions"]
               for low, middle, high in zip(prediction["0.1"], prediction["0.5"], prediction["0.9"]))


def check_predictions(actual, expected, batch_size, horizon=HORIZON):
    actual_values = prediction_array(actual, batch_size, horizon)
    expected_values = prediction_array(expected, batch_size, horizon)
    maximum = max(abs(first - second) for first, second in zip(actual_values, expected_values))
    if not all(math.isclose(first, second, rel_tol=1e-4, abs_tol=1e-4)
               for first, second in zip(actual_values, expected_values)):
        raise AssertionError({"max_abs_error": maximum, "rtol": 1e-4, "atol": 1e-4})
    actual_ids = [item.get("item_id") for item in actual["data"]["predictions"]]
    expected_ids = [item.get("item_id") for item in expected["data"]["predictions"]]
    if actual_ids != expected_ids:
        raise AssertionError("Prediction item IDs/order differ")
    return {
        "max_abs_error": maximum, "rtol": 1e-4, "atol": 1e-4,
        "values_compared": len(actual_values), "passed": True,
        "native_quantile_crossings": quantile_crossings(expected),
        "actual_quantile_crossings": quantile_crossings(actual),
    }


def payload_for(data_path, batch_size, context):
    import numpy as np
    from vtsfm.config import MODEL_ID

    if batch_size <= 0 or context <= 0:
        raise ValueError("batch_size and context must be positive")
    with np.load(data_path, allow_pickle=False) as data:
        series = data["series"]
    if series.ndim != 2 or not len(series) or series.shape[1] < context + HORIZON:
        raise ValueError(f"Data must contain channels × at least {context + HORIZON} positions")
    if not np.isfinite(series).all():
        raise ValueError("Prepared data contains nonfinite values")
    available = series.shape[1] - context - HORIZON + 1
    inputs = []
    for index in range(batch_size):
        channel = index % len(series)
        start = ((index // len(series)) * 37) % available
        target = series[channel, start:start + context].astype(np.float32).tolist()
        inputs.append({"target": target, "item_id": f"channel-{channel}-offset-{start}"})
    body = {"model": MODEL_ID, "task": "plugin", "data": {
        "inputs": inputs, "parameters": {
            "prediction_length": HORIZON, "quantile_levels": QUANTILES,
            "cross_learning": False,
        }}}
    encoded = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    return body, encoded, hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
