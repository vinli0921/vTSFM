"""Incomplete runs must never become complete report cells."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.report import generate_report, load_results


class BenchmarkReportTests(unittest.TestCase):
    def test_partial_and_missing_cells_stay_visible_and_backend_is_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            results.mkdir()
            (results / "config.json").write_text(json.dumps({"grid": [[1, 128]],
                "blocks": 2, "repetitions": 3, "plugin_backend": "vtsfm", "synthetic": True}))
            (results / "points.jsonl").write_text(json.dumps({"engine": "native", "batch_size": 1,
                "context": 128, "block": 0, "http_latencies_s": [0.1, 0.2],
                "parity": {"passed": True}, "timed_parity_max_abs": 0,
                "payload_sha256": "example"}) + "\n")
            data = load_results(results)
            self.assertEqual(data["cells"][("native", 1, 128)]["status"], "partial")
            self.assertEqual(data["cells"][("vllm", 1, 128)]["status"], "missing")
            outcome = generate_report(results, Path(temporary) / "report", plots=False)
            self.assertEqual(outcome["complete_cells"], 0)
            report = (Path(temporary) / "report" / "report.md").read_text()
            self.assertIn("vtsfm", report)
            self.assertIn("Synthetic smoke", report)
            self.assertNotIn("Original service restoration", report)
