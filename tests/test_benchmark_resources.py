"""Resource refusal and process ownership must work without a GPU."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

from benchmarks.run import (OwnedProcesses, backend_command, ensure_gpu_idle,
                            ensure_port_free, ensure_worker_gpu, measure_point, parse_args)


class BenchmarkResourceTests(unittest.TestCase):
    def test_selected_gpu_uses_numeric_visibility_supported_by_pinned_vllm(self):
        for backend in ("native", "vtsfm", "chronos2"):
            with self.subTest(backend=backend), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-parent-uuid"}):
                _, environment = backend_command(backend, port=8001, gpu_index=3)
                self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
                self.assertEqual(int(environment["CUDA_VISIBLE_DEVICES"]), 3)
                self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_worker_gpu_identity_accepts_torch_uuid_format_and_rejects_wrong_device(self):
        ensure_worker_gpu({"gpu": {"uuid": "abcdef-1234"}}, "GPU-ABCDEF-1234")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            ensure_worker_gpu({"gpu": {"uuid": "wrong-device"}}, "GPU-abcdef-1234")

    def test_strict_worker_gpu_identity_rejects_missing_cuda_or_nvml_uuid(self):
        for worker, device in ((None, "GPU-abc"), ("abc", "")):
            with self.subTest(worker=worker, device=device), self.assertRaisesRegex(RuntimeError, "without a UUID"):
                ensure_worker_gpu({"gpu": {"uuid": worker}}, device, require_uuid=True)

    def test_an_occupied_port_is_refused(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "port"):
                ensure_port_free("127.0.0.1", listener.getsockname()[1])

    def test_any_existing_compute_process_is_refused(self):
        class FakeNvml:
            @staticmethod
            def nvmlDeviceGetComputeRunningProcesses(handle):
                return [type("Process", (), {"pid": 12345})()]
            @staticmethod
            def nvmlDeviceGetGraphicsRunningProcesses(handle):
                return []
        with self.assertRaisesRegex(RuntimeError, "12345"):
            ensure_gpu_idle(FakeNvml, object())

    def test_context_cleanup_reaps_only_owned_children_on_failure(self):
        external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with tempfile.TemporaryDirectory() as directory:
                child = None
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with OwnedProcesses(Path(directory)) as owner:
                        with self.assertRaises(ValueError):
                            owner.stop(external)
                        child = owner.spawn([sys.executable, "-c", "import time; time.sleep(60)"],
                                            os.environ.copy(), "test")
                        raise RuntimeError("intentional")
                self.assertIsNotNone(child.poll())
                self.assertIsNone(external.poll())
        finally:
            external.terminate()
            external.wait(timeout=5)

    def test_smoke_defaults_and_backend_constraints_are_explicit(self):
        args = parse_args(["--smoke", "--synthetic", "--backends", "native", "chronos2"])
        self.assertEqual(args.grid, [(2, 128)])
        self.assertEqual((args.blocks, args.repetitions, args.warmups), (1, 3, 2))
        self.assertEqual(args.backends, ["native", "chronos2"])
        with self.assertRaises(SystemExit):
            parse_args(["--synthetic"])
        with self.assertRaises(SystemExit):
            parse_args(["--backends", "vtsfm", "chronos2"])

    def test_interrupt_preserves_completed_latencies_and_reaps_owned_child(self):
        response = {"data": {"predictions": [{"mean": [1.0]*96,
                    "0.1": [.5]*96, "0.5": [1.0]*96, "0.9": [1.5]*96}]}}
        monitor = SimpleNamespace(used=lambda: 1234)
        args = SimpleNamespace(warmups=0, repetitions=3, inprocess_repetitions=2)
        for interruption in (KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(interruption=type(interruption).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                reference = output / "references" / "b1-c128.json"
                reference.parent.mkdir()
                child = None
                with self.assertRaises(type(interruption)):
                    with OwnedProcesses(output) as owner:
                        child = owner.spawn([sys.executable, "-c", "import time; time.sleep(60)"],
                                            os.environ.copy(), "cancel-test")
                        with patch("benchmarks.run.memory", return_value={}), patch(
                            "benchmarks.run.forecast", side_effect=[(response, .01), (response, .02), interruption]
                        ):
                            measure_point(None, "unused", "native", 0, 1, 128, b"{}", "digest",
                                          reference, monitor, 1234, args,
                                          record_path=output / "points.jsonl")
                self.assertIsNotNone(child.poll())
                record = json.loads((output / "points.jsonl").read_text())
                self.assertEqual(record["http_latencies_s"], [.02])
                self.assertTrue(record["interrupted"])
                self.assertGreater(len(record["nvml_samples"]), 0)
