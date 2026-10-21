"""CPU contracts for published benchmark statistics and numerical parity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from benchmarks.common import check_predictions, payload_for, summarize


class BenchmarkMetricsTests(unittest.TestCase):
    def test_percentiles_and_throughput_follow_recorded_requests(self):
        row = summarize([0.01, 0.02, 0.03, 0.04], batch_size=4)
        self.assertAlmostEqual(row["p50_ms"], 25.0)
        self.assertAlmostEqual(row["p95_ms"], 38.5)
        self.assertAlmostEqual(row["request_per_s"], 40.0)
        self.assertAlmostEqual(row["series_per_s"], 160.0)
        for invalid in ([], [0.0], [float("nan")], [float("inf")]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize(invalid, 4)

    def test_payload_is_reproducible_and_covers_actual_numeric_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.npz"
            np.savez(path, series=np.arange(2 * 256, dtype=np.float32).reshape(2, 256))
            first = payload_for(path, 4, 128)
            second = payload_for(path, 4, 128)
            self.assertEqual(first, second)
            body, encoded, digest = first
            self.assertEqual(json.loads(encoded), body)
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
            self.assertEqual(len(body["data"]["inputs"]), 4)
            self.assertEqual(body["data"]["inputs"][0]["target"], list(range(128)))
            self.assertNotEqual(digest, payload_for(path, 2, 128)[2])
            with self.assertRaises(ValueError):
                payload_for(path, 4, 200)

    def test_every_quantile_value_is_checked_not_only_point_forecast(self):
        response = {"data": {"predictions": [{"mean": [1.0] * 96,
                    "0.1": [0.5] * 96, "0.5": [1.0] * 96, "0.9": [1.5] * 96}]}}
        actual = json.loads(json.dumps(response))
        actual["data"]["predictions"][0]["0.9"][-1] = 9.0
        with self.assertRaises(AssertionError):
            check_predictions(actual, response, 1)
        self.assertEqual(check_predictions(response, response, 1)["values_compared"], 384)

    def test_parity_is_enforced_with_python_optimization(self):
        code = '''
from benchmarks.common import check_predictions
expected = {"data": {"predictions": [{"item_id": "expected", "mean": [1.0]*96,
    "0.1": [.5]*96, "0.5": [1.0]*96, "0.9": [1.5]*96}]}}
actual = {"data": {"predictions": [{"item_id": "wrong", "mean": [10.0]*96,
    "0.1": [.5]*96, "0.5": [1.0]*96, "0.9": [1.5]*96}]}}
try:
    check_predictions(actual, expected, 1)
except (AssertionError, ValueError):
    pass
else:
    raise SystemExit("Optimized Python incorrectly accepted divergent forecasts")
'''
        environment = os.environ.copy()
        environment["PYTHONOPTIMIZE"] = "1"
        process = subprocess.run([sys.executable, "-O", "-c", code],
                                 env=environment, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
