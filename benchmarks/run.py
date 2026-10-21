"""Compare native HTTP with one plugin using only backend processes we launch.

Prepare data explicitly with ``python -m benchmarks.prepare_data --output data``.
For a download-free integration check use ``--smoke --synthetic``. Synthetic
smoke measurements are labelled and must not be used as performance evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

try:
    from .common import HORIZON, QUANTILES, check_predictions, payload_for, summarize, write_json
except ImportError:  # direct script execution
    from common import HORIZON, QUANTILES, check_predictions, payload_for, summarize, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]


def emit(event, **fields):
    print(json.dumps({"event": event, "time": time.time(), **fields}), flush=True)


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def ensure_port_free(host, port):
    """Fail before launching anything; never attach to or stop an existing server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Uvicorn uses SO_REUSEADDR too: closed connections in TIME_WAIT do not
        # occupy a listening port between our sequential backend processes.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            raise RuntimeError(f"Benchmark port {host}:{port} is unavailable: {error}") from error


def ensure_gpu_idle(nvml, gpu):
    """Refuse compute/graphics occupants; do not kill them or silently reuse them."""
    processes = list(nvml.nvmlDeviceGetComputeRunningProcesses(gpu))
    try:
        processes += list(nvml.nvmlDeviceGetGraphicsRunningProcesses(gpu))
    except Exception as error:
        unsupported = getattr(nvml, "NVMLError_NotSupported", ())
        if not unsupported or not isinstance(error, unsupported):
            raise
    pids = sorted({process.pid for process in processes})
    if pids:
        raise RuntimeError(f"Selected GPU is occupied by processes {pids}; use an idle GPU")


class OwnedProcesses:
    """Keep explicit ownership; arbitrary PIDs or pre-existing processes are rejected."""

    def __init__(self, output):
        self.output = Path(output)
        self.children = {}
        self.history = []

    def __enter__(self):
        return self

    def spawn(self, command, environment, label):
        with (self.output / f"{label}.log").open("ab", buffering=0) as logfile:
            child = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.children[child.pid] = child
        self.history.append({"pid": child.pid, "label": label, "command": command})
        write_json(self.output / "owned_processes.json", self.history)
        return child

    def stop(self, child):
        if self.children.get(child.pid) is not child:
            raise ValueError("Refusing to stop a process that this runner does not own")
        try:
            # Each owned backend has its own session; this includes its workers.
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
        # A backend parent can exit before its workers. Finish that owned group.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        del self.children[child.pid]

    def close(self):
        errors = []
        for child in list(self.children.values()):
            try:
                self.stop(child)
            except Exception as error:
                errors.append(repr(error))
        write_json(self.output / "cleanup.json", {
            "all_stopped": not self.children, "errors": errors,
            "owned_pids": [entry["pid"] for entry in self.history], "time": time.time(),
        })
        if errors:
            raise RuntimeError(f"Owned backend cleanup failed: {errors}")

    def __exit__(self, exception_type, exception, traceback_object):
        self.close()


class GpuMonitor:
    def __init__(self, index):
        import pynvml

        self.nvml = pynvml
        self.nvml.nvmlInit()
        try:
            self.gpu = self.nvml.nvmlDeviceGetHandleByIndex(index)
            self.index = index
            self.uuid = self.nvml.nvmlDeviceGetUUID(self.gpu)
            if isinstance(self.uuid, bytes):
                self.uuid = self.uuid.decode()
        except BaseException:
            self.nvml.nvmlShutdown()
            raise

    def assert_idle(self):
        ensure_gpu_idle(self.nvml, self.gpu)

    def wait_idle(self, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.assert_idle()
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)

    def used(self):
        return self.nvml.nvmlDeviceGetMemoryInfo(self.gpu).used

    def metadata(self):
        def decode(value):
            return value.decode() if isinstance(value, bytes) else value
        return {
            "index": self.index, "uuid": self.uuid,
            "name": decode(self.nvml.nvmlDeviceGetName(self.gpu)),
            "driver": decode(self.nvml.nvmlSystemGetDriverVersion()),
            "total_memory_bytes": self.nvml.nvmlDeviceGetMemoryInfo(self.gpu).total,
        }

    def close(self):
        self.nvml.nvmlShutdown()


class Sampler:
    def __init__(self, monitor, interval=0.02):
        self.monitor = monitor
        self.interval = interval
        self.samples = []
        self.done = threading.Event()
        self.error = None

    def sample(self):
        try:
            while not self.done.is_set():
                self.samples.append([time.time(), self.monitor.used()])
                self.done.wait(self.interval)
        except BaseException as error:
            self.error = error

    def __enter__(self):
        self.samples.append([time.time(), self.monitor.used()])
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.done.set()
        self.thread.join(timeout=2)
        self.samples.append([time.time(), self.monitor.used()])
        if self.error is not None:
            raise RuntimeError("GPU memory sampler failed") from self.error
        if self.thread.is_alive():
            raise RuntimeError("GPU memory sampler did not stop")


def backend_command(backend, port, gpu_index):
    from vtsfm.cli import build_vllm_command

    if backend == "native":
        environment = os.environ.copy()
        environment.pop("VLLM_PLUGINS", None)
        environment.pop("VLLM_SERVER_DEV_MODE", None)
        command = [sys.executable, "-m", "uvicorn", "vtsfm.native_server:app",
                   "--host", "127.0.0.1", "--port", str(port),
                   "--workers", "1", "--log-level", "warning"]
    else:
        command, environment = build_vllm_command(
            backend=backend, host="127.0.0.1", port=port, benchmark=True)
    # vLLM 0.16 parses visible device IDs as integers; UUID-valued visibility
    # fails before model loading. Preserve the selected physical index instead.
    environment.update(OMP_NUM_THREADS="4", HF_HUB_DISABLE_TELEMETRY="1",
                       CUDA_VISIBLE_DEVICES=str(gpu_index), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return command, environment


def ensure_worker_gpu(worker_config, selected_uuid):
    worker_uuid = worker_config.get("gpu", {}).get("uuid")
    # Torch reports a bare UUID; NVML includes the GPU- prefix.
    normalize = lambda value: str(value).lower().removeprefix("gpu-")
    if worker_uuid and normalize(worker_uuid) != normalize(selected_uuid):
        raise RuntimeError(f"Backend GPU UUID {worker_uuid} does not match selected NVML GPU {selected_uuid}")


def wait_ready(child, url, session, timeout):
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Owned backend exited {child.returncode}; inspect its log")
        try:
            if session.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Owned backend did not become healthy within {timeout}s")


def memory(session, url, engine, action):
    if engine == "vllm":
        response = session.post(url + "/collective_rpc", json={
            "method": "bench_memory_" + action, "args": [], "kwargs": {}, "timeout": 30}, timeout=45)
        response.raise_for_status()
        results = response.json()["results"]
        if len(results) != 1:
            raise RuntimeError("This single-GPU benchmark expects exactly one worker memory result")
        return results[0]
    if action == "config":
        response = session.get(url + "/bench/config", timeout=45)
    else:
        response = session.post(url + "/bench/memory/" + action, timeout=45)
    response.raise_for_status()
    return response.json()


def forecast(session, url, encoded):
    started = time.perf_counter()
    response = session.post(url + "/pooling", data=encoded,
                            headers={"Content-Type": "application/json"}, timeout=180)
    response.raise_for_status()
    body = response.json()
    return body, time.perf_counter() - started


def append(path, value):
    with Path(path).open("a") as target:
        target.write(json.dumps(value, allow_nan=False) + "\n")


def build_summary(output):
    grouped = {}
    for line in (output / "points.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("http_latencies_s"):
            grouped.setdefault((record["engine"], record["batch_size"], record["context"]), []).append(record)
    rows = []
    for (engine, batch, context), records in sorted(grouped.items()):
        latencies = [value for record in records for value in record["http_latencies_s"]]
        internal = [value for record in records for value in record.get("inprocess_latencies_s", [])]
        row = {"engine": engine, "batch_size": batch, "context": context,
               **summarize(latencies, batch), "blocks": len(records),
               "native_inprocess_n": len(internal),
               "native_inprocess_p50_ms": summarize(internal, batch)["p50_ms"] if internal else None,
               "payload_sha256": records[0]["payload_sha256"]}
        for field, location in (
            ("loaded_gpu_mib", ("loaded_gpu_bytes",)),
            ("warmed_gpu_mib", ("warmed_gpu_bytes",)),
            ("sampled_peak_gpu_mib", ("sampled_peak_gpu_bytes",)),
            ("peak_allocated_mib", ("memory_after", "peak_allocated_bytes")),
            ("peak_reserved_mib", ("memory_after", "peak_reserved_bytes")),
        ):
            values = []
            for record in records:
                value = record
                for part in location:
                    value = value.get(part) if isinstance(value, dict) else None
                if isinstance(value, (int, float)):
                    values.append(value / 2**20)
            row[field] = max(values) if values else None
        rows.append(row)
    write_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w", newline="") as target:
        fields = list(rows[0]) if rows else ["engine", "batch_size", "context", "n"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use labelled synthetic inputs; requires --smoke")
    parser.add_argument("--output", type=Path, default=Path("results") / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prepare-data", action="store_true", help="Explicitly allow fetching/preparing the pinned fixture")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gpu", type=int, default=0, help="Physical NVML GPU index; children see only this GPU")
    parser.add_argument("--backends", nargs=2, default=["native", "vtsfm"], metavar=("NATIVE", "PLUGIN"))
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--contexts", type=int, nargs="+")
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--inprocess-repetitions", type=int)
    parser.add_argument("--startup-timeout", type=float, default=240)
    args = parser.parse_args(argv)
    if args.backends[0] != "native" or args.backends[1] not in ("vtsfm", "chronos2"):
        parser.error("--backends must be native vtsfm or native chronos2")
    if args.synthetic and not args.smoke:
        parser.error("--synthetic requires --smoke; generated inputs are not benchmark data")
    if not 1 <= args.port <= 65535 or args.gpu < 0 or args.startup_timeout <= 0:
        parser.error("port, GPU index, and startup timeout are invalid")
    defaults = {"blocks": 1 if args.smoke else 3, "repetitions": 3 if args.smoke else 50,
                "warmups": 2 if args.smoke else 10, "inprocess_repetitions": 2 if args.smoke else 10}
    for name, default in defaults.items():
        value = getattr(args, name)
        setattr(args, name, default if value is None else value)
        if getattr(args, name) < (0 if name == "warmups" else 1):
            parser.error(f"--{name.replace('_', '-')} is out of range")
    batches = args.batches or ([2] if args.smoke else [1, 4, 16, 64, 128])
    contexts = args.contexts or ([128] if args.smoke else [128, 512, 2048, 4096])
    if any(value <= 0 for value in batches + contexts):
        parser.error("batches and contexts must be positive")
    args.grid = [(batch, context) for batch in sorted(set(batches)) for context in sorted(set(contexts))]
    return args


def prepare_inputs(args, output):
    import numpy as np

    data_dir = resolve_path(args.data_dir)
    if args.synthetic:
        data_dir = output / "synthetic-smoke-data"
        data_dir.mkdir()
        length = max(context for _, context in args.grid) + HORIZON + 128
        positions = np.arange(length, dtype=np.float32)
        series = np.stack([np.sin(positions / (11 + channel)) + channel for channel in range(7)]).astype(np.float32)
        np.savez_compressed(data_dir / "data.npz", series=series)
        manifest = {"dataset_name": "Synthetic smoke inputs (not observational data)",
                    "synthetic": True, "array": {"shape": list(series.shape), "dtype": "float32"},
                    "generator": "deterministic channel-offset sine waves; integration checks only"}
        manifest["array"]["sha256"] = hashlib.sha256((data_dir / "data.npz").read_bytes()).hexdigest()
        write_json(data_dir / "data_manifest.json", manifest)
    else:
        if args.prepare_data:
            try:
                from .prepare_data import prepare
            except ImportError:
                from prepare_data import prepare
            prepare(data_dir)
        if not (data_dir / "data.npz").is_file() or not (data_dir / "data_manifest.json").is_file():
            raise RuntimeError("Prepared data is missing. Run python -m benchmarks.prepare_data --output data, "
                               "pass --prepare-data explicitly, or use --smoke --synthetic.")
        manifest = json.loads((data_dir / "data_manifest.json").read_text())
        digest = hashlib.sha256((data_dir / "data.npz").read_bytes()).hexdigest()
        if digest != manifest.get("array", {}).get("sha256"):
            raise ValueError("Prepared data checksum does not match its provenance manifest")
    for batch, context in args.grid:
        payload_for(data_dir / "data.npz", batch, context)
    return data_dir / "data.npz", manifest


def measure_point(session, url, engine, block, batch, context, encoded, digest,
                  reference_path, monitor, loaded, args, record_path):
    record = {"engine": engine, "block": block, "batch_size": batch, "context": context,
              "payload_sha256": digest, "payload_bytes": len(encoded), "http_latencies_s": [],
              "loaded_gpu_bytes": loaded}
    try:
        memory(session, url, engine, "empty_cache")
        initial, _ = forecast(session, url, encoded)
        if engine == "native" and block == 0:
            # Validate every output, including shape, before saving a reference.
            check_predictions(initial, initial, batch)
            write_json(reference_path, initial)
        if not reference_path.exists():
            raise RuntimeError("Native correctness reference is missing")
        expected = json.loads(reference_path.read_text())
        record["parity"] = check_predictions(initial, expected, batch)
        for _ in range(args.warmups):
            forecast(session, url, encoded)
        record["warmed_gpu_bytes"] = monitor.used()
        record["memory_before"] = memory(session, url, engine, "reset")
        timed_responses = []
        sampler = Sampler(monitor)
        try:
            with sampler:
                for _ in range(args.repetitions):
                    response, elapsed = forecast(session, url, encoded)
                    record["http_latencies_s"].append(elapsed)
                    timed_responses.append(response)
        finally:
            # Save completed samples even when SIGINT/SIGTERM interrupts a request.
            record["nvml_samples"] = sampler.samples
            if sampler.samples:
                record["sampled_peak_gpu_bytes"] = max(value for _, value in sampler.samples)
        record["memory_after"] = memory(session, url, engine, "stats")
        # Numerical checks of all forecasts happen after all timed intervals.
        record["timed_parity"] = [check_predictions(response, expected, batch) for response in timed_responses]
        record["timed_parity_checks"] = len(record["timed_parity"])
        record["timed_parity_max_abs"] = max(item["max_abs_error"] for item in record["timed_parity"])
        del timed_responses
        if engine == "native":
            body = json.loads(encoded)
            body["repetitions"] = args.inprocess_repetitions
            response = session.post(url + "/bench/inprocess", json=body, timeout=180)
            response.raise_for_status()
            record["inprocess_latencies_s"] = response.json()["latencies_s"]
        record["completed_time"] = time.time()
        return record, None
    except Exception as error:
        record["error"] = repr(error)
        failure = {key: record[key] for key in ("engine", "block", "batch_size", "context")}
        failure.update(error=repr(error), traceback=traceback.format_exc())
        return record, failure
    except BaseException as interruption:
        record["error"] = repr(interruption)
        record["interrupted"] = True
        raise
    finally:
        # This is the only writer for this point, on success, failure, or cancellation.
        record["recorded_time"] = time.time()
        append(record_path, record)


def main(argv=None):
    args = parse_args(argv)
    # Do this before importing GPU dependencies or creating a backend process.
    ensure_port_free("127.0.0.1", args.port)
    import requests
    from vtsfm.config import CHRONOS_COMMIT, MODEL_ID, MODEL_REVISION

    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "references").mkdir()
    (output / "harness").mkdir()
    for source in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(source, output / "harness" / source.name)
    for name in ("native_server.py", "benchmark_memory.py"):
        source = REPO_ROOT / "src" / "vtsfm" / name
        if source.exists():
            shutil.copy2(source, output / "harness" / name)
    data_path, manifest = prepare_inputs(args, output)
    plugin = args.backends[1]
    config = vars(args).copy()
    config.update(output=str(output), data_dir=str(data_path.parent),
                  plugin_backend=plugin, engine_labels={"native": "native", "vllm": plugin},
                  forecast_horizon=HORIZON, dtype="float32", quantiles=QUANTILES,
                  cross_learning=False, model=MODEL_ID, revision=MODEL_REVISION,
                  nvml_interval_s=0.02, one_outstanding_request=True,
                  http_boundary="preencoded request submission through JSON response decoding",
                  native_inprocess_repetitions_per_block=args.inprocess_repetitions)
    write_json(output / "config.json", config)
    write_json(output / "metadata.json", {
        "start_time": time.time(), "hostname": platform.node(), "logical_cpus": os.cpu_count(),
        "installed_packages": {distribution.metadata["Name"]: distribution.version
                               for distribution in importlib.metadata.distributions()},
        "chronos_git_commit": CHRONOS_COMMIT, "source_manifest": manifest,
    })
    monitor = GpuMonitor(args.gpu)
    errors = []
    invalid = set()
    url = f"http://127.0.0.1:{args.port}"
    try:
        monitor.assert_idle()
        metadata = json.loads((output / "metadata.json").read_text())
        metadata["gpu"] = monitor.metadata()
        write_json(output / "metadata.json", metadata)
        with OwnedProcesses(output) as owner:
            for block in range(args.blocks):
                order = ["native", plugin] if block % 2 == 0 else [plugin, "native"]
                points = list(args.grid)
                random.Random(20261020 + block).shuffle(points)
                if block == 0:
                    largest = max(points, key=lambda point: point[0] * point[1])
                    points.remove(largest)
                    points.insert(0, largest)
                for backend in order:
                    engine = "native" if backend == "native" else "vllm"
                    ensure_port_free("127.0.0.1", args.port)
                    monitor.assert_idle()
                    command, environment = backend_command(backend, args.port, monitor.index)
                    child = owner.spawn(command, environment, f"{engine}-block-{block}")
                    try:
                        with requests.Session() as session:
                            session.trust_env = False  # loopback benchmark bypasses HTTP proxies
                            wait_ready(child, url, session, args.startup_timeout)
                            loaded = monitor.used()
                            worker_config = memory(session, url, engine, "config")
                            write_json(output / f"{engine}-config-block-{block}.json", worker_config)
                            ensure_worker_gpu(worker_config, monitor.uuid)
                            emit("backend_ready", backend=backend, block=block, pid=child.pid)
                            for batch, context in points:
                                if (engine, batch, context) in invalid:
                                    continue
                                _, encoded, digest = payload_for(data_path, batch, context)
                                record, failure = measure_point(
                                    session, url, engine, block, batch, context, encoded, digest,
                                    output / "references" / f"b{batch}-c{context}.json",
                                    monitor, loaded, args, record_path=output / "points.jsonl")
                                if failure:
                                    errors.append(failure)
                                    invalid.add((engine, batch, context))
                                    append(output / "errors.jsonl", failure)
                                    emit("point_failed", **failure)
                                else:
                                    emit("point_complete", engine=engine, backend=backend,
                                         block=block, batch=batch, context=context,
                                         **summarize(record["http_latencies_s"], batch))
                    finally:
                        owner.stop(child)
                        monitor.wait_idle()
    except BaseException as error:
        append(output / "errors.jsonl", {"error": repr(error), "scope": "run", "traceback": traceback.format_exc()})
        raise
    finally:
        monitor.close()
        if (output / "points.jsonl").exists():
            build_summary(output)
    if errors:
        raise RuntimeError(f"{len(errors)} workload cells failed; inspect {output / 'errors.jsonl'}")
    emit("run_complete", output=str(output), plugin_backend=plugin)
    return output


if __name__ == "__main__":
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
