"""Local-only official Chronos-2 baseline with the plugin's pooling payload.

Run with the benchmark environment's Python; defaults to 127.0.0.1:8001 and
exactly one Uvicorn worker. OMP_NUM_THREADS and Torch precision flags are inherited.
"""

import argparse
from contextlib import asynccontextmanager
import importlib.metadata
import os
import platform
import threading
import time
from typing import Any

import torch
from chronos import Chronos2Pipeline
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from vtsfm.benchmark_memory import MemoryMetrics
from vtsfm.adapters.chronos2.spec import MODEL_ID, MODEL_REVISION
from vtsfm.config import DEFAULT_PORT


DEVICE = torch.device("cuda:0")
_lock = threading.Lock()
_pipeline: Chronos2Pipeline | None = None


class NativeMemoryMetrics(MemoryMetrics):
    device = DEVICE


_memory = NativeMemoryMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("The native benchmark requires CUDA")
    _pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="cuda:0",
        torch_dtype=torch.float32,
    )
    _pipeline.model.eval()
    torch.cuda.synchronize(DEVICE)
    yield


app = FastAPI(lifespan=lifespan)


def _parse_payload(payload: dict[str, Any]):
    data = payload.get("data")
    if not isinstance(data, dict):
        raise HTTPException(422, "data must be an object")
    inputs = data.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise HTTPException(422, "data.inputs must be a nonempty list")
    if any(not isinstance(item, dict) or "target" not in item for item in inputs):
        raise HTTPException(422, "Each input must contain target")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        raise HTTPException(422, "data.parameters must be an object")
    if parameters.get("cross_learning", False) is not False:
        raise HTTPException(422, "This baseline requires cross_learning=false")
    horizon = parameters.get("prediction_length", 1)
    if type(horizon) is not int or not 1 <= horizon <= 1024:
        raise HTTPException(422, "prediction_length must be an integer in [1,1024]")
    levels = parameters.get("quantile_levels", [0.1, 0.5, 0.9])
    if (
        not isinstance(levels, list)
        or not levels
        or any(type(q) not in (int, float) or not 0 < q < 1 for q in levels)
        or levels != sorted(set(levels))
    ):
        raise HTTPException(422, "quantile_levels must be unique, increasing values in (0,1)")
    return inputs, horizon, [float(q) for q in levels]


def _tensor_to_list(tensor: torch.Tensor):
    tensor = tensor.detach().cpu()
    if tensor.ndim != 2:
        raise RuntimeError(f"Expected [variates,horizon], received {tuple(tensor.shape)}")
    return tensor[0].tolist() if tensor.shape[0] == 1 else tensor.tolist()


def _forecast(payload: dict[str, Any]):
    if _pipeline is None:
        raise HTTPException(503, "Model is not loaded")
    inputs, horizon, levels = _parse_payload(payload)
    try:
        targets = [torch.tensor(item["target"], dtype=torch.float32) for item in inputs]
    except (TypeError, ValueError) as error:
        raise HTTPException(422, f"Invalid target: {error}") from error
    if any(target.ndim not in (1, 2) or target.numel() == 0 for target in targets):
        raise HTTPException(422, "Each target must be a nonempty 1-D or 2-D array")
    with torch.inference_mode():
        quantiles, means = _pipeline.predict_quantiles(
            inputs=targets,
            prediction_length=horizon,
            quantile_levels=levels,
            batch_size=len(inputs),
            cross_learning=False,
        )
    if len(quantiles) != len(inputs) or len(means) != len(inputs):
        raise RuntimeError("Native pipeline returned the wrong prediction count")
    predictions = []
    for item, quantile_tensor, mean_tensor in zip(inputs, quantiles, means):
        prediction = {"mean": _tensor_to_list(mean_tensor)}
        for level, values in zip(levels, quantile_tensor.unbind(dim=-1)):
            prediction[str(level)] = _tensor_to_list(values)
        if item.get("item_id") is not None:
            prediction["item_id"] = item["item_id"]
        predictions.append(prediction)
    return {"data": {"predictions": predictions}}


@app.get("/health")
def health():
    if _pipeline is None:
        raise HTTPException(503, "Model is not loaded")
    return {"status": "ok"}


@app.post("/pooling")
def pooling(payload: dict[str, Any] = Body(...)):
    with _lock:
        return JSONResponse(content=_forecast(payload))


@app.post("/bench/memory/empty_cache")
def empty_cache():
    with _lock:
        return _memory.bench_memory_empty_cache()


@app.post("/bench/memory/reset")
def reset_memory():
    with _lock:
        return _memory.bench_memory_reset()


@app.post("/bench/memory/stats")
def memory_stats():
    with _lock:
        return _memory.bench_memory_stats()


@app.post("/bench/inprocess")
def inprocess(payload: dict[str, Any] = Body(...)):
    repetitions = payload.get("repetitions", 10)
    if type(repetitions) is not int or not 1 <= repetitions <= 10000:
        raise HTTPException(422, "repetitions must be an integer in [1,10000]")
    _parse_payload(payload)
    latencies = []
    with _lock:
        for _ in range(repetitions):
            torch.cuda.synchronize(DEVICE)
            started = time.perf_counter()
            response = _forecast(payload)
            torch.cuda.synchronize(DEVICE)
            latencies.append(time.perf_counter() - started)
            del response
    return {"latencies_s": latencies}


def _version(distribution: str):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@app.get("/bench/config")
def config():
    with _lock:
        properties = torch.cuda.get_device_properties(DEVICE)
        model = _pipeline.model if _pipeline is not None else None
        return {
            "backend": "official_chronos2_pipeline",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dtype": str(next(model.parameters()).dtype) if model is not None else None,
            "cross_learning": False,
            "batch_size_policy": "len(inputs)",
            "context_length": _pipeline.model_context_length if _pipeline is not None else None,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                **{name: _version(name) for name in (
                    "chronos-forecasting", "transformers", "accelerate", "fastapi", "uvicorn", "vllm"
                )},
            },
            "gpu": {
                "device": str(DEVICE),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "uuid": str(getattr(properties, "uuid", "")),
            },
            "torch_settings": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
            },
            "environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"
            )},
            "pid": os.getpid(),
        }


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
