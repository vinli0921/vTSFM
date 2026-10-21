"""The CPU preparer preserves causality and rejects changed source bytes."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.prepare_data import COLUMNS, fetch_source, prepare


class BenchmarkDataTests(unittest.TestCase):
    def test_unpinned_source_bytes_are_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text("[]")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch_source(source, Path(directory) / "unused-cache.json")

    def test_missing_rows_use_previous_observation_without_future_leakage(self):
        # Artificial fixture only for this unit test; never benchmark observations.
        beginning = datetime(2020, 2, 1)
        rows = []
        for index in range(5116):
            row = {"asset_id": "mp_1", "timestamp": (beginning + timedelta(hours=index)).isoformat()}
            row.update({name: ("" if 10 <= index < 570 else index + channel)
                        for channel, (name, _, _) in enumerate(COLUMNS)})
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("benchmarks.prepare_data.fetch_source", return_value=json.dumps(rows).encode()):
                manifest = prepare(output)
            with np.load(output / "data.npz") as data:
                series = data["series"]
            self.assertEqual(series.shape, (7, 5116))
            np.testing.assert_array_equal(series[:, 10], series[:, 9])
            np.testing.assert_array_equal(series[:, 569], series[:, 9])
            self.assertEqual(float(series[0, 570]), 570.0)
            self.assertEqual(manifest["imputation"]["imputed_rows"], 560)
            self.assertEqual(manifest["synthetically_generated_observations"], 0)
