"""Manual native HTTP baseline for TimesFM, Toto, and Moirai integration checks.

Launch one worker on loopback with the selected adapter environment::

    VTSFM_TEST_ADAPTER=timesfm25 python -m uvicorn \
        native_forecast_server:create_app --factory --host 127.0.0.1 --port 8001

Set PYTHONPATH to this checkout's tests, src, and root directories. Imports and
the app factory are CPU-safe; startup alone loads the official CUDA FP32 model.
"""

from contextlib import asynccontextmanager
import os
import threading
import time
from uuid import uuid4

from model_integration_support import NativeForecaster, forecast_settings


def create_app():
    from fastapi import Body, FastAPI, HTTPException
    from vtsfm.adapters import get_adapter
    from vtsfm.adapters.forecast import format_predictions, parse_request

    adapter = os.environ.get('VTSFM_TEST_ADAPTER', '')
    settings = forecast_settings(adapter)
    spec = get_adapter(adapter)
    lock = threading.Lock()
    forecaster = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal forecaster
        forecaster = NativeForecaster(adapter)
        try:
            yield
        finally:
            forecaster = None

    app = FastAPI(lifespan=lifespan)

    def ready():
        if forecaster is None:
            raise HTTPException(503, 'Model is not loaded')
        return forecaster

    @app.get('/health')
    def health():
        ready()
        return {'status': 'ok'}

    @app.get('/metadata')
    def metadata():
        return ready().metadata()

    @app.post('/pooling')
    def pooling(payload: dict = Body(...)):
        model = ready()
        if set(payload) != {'model', 'task', 'data'}:
            raise HTTPException(422, 'Expected exactly model, task, and data')
        if payload['model'] != spec.model_id or payload['task'] != 'plugin':
            raise HTTPException(422, 'Expected the selected model and task=plugin')
        try:
            inputs, horizon, levels = parse_request(payload['data'], settings)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        # Sync routes run in FastAPI's thread pool. One official forecast may
        # execute at a time, matching the current vLLM max_num_seqs=1 baseline.
        try:
            with lock:
                values = model.predict_array(payload['data'])
            data = format_predictions(values, [item.get('item_id') for item in inputs],
                                      levels, horizon, settings)
        except ValueError as error:
            raise HTTPException(500, str(error)) from error
        return {'request_id': str(uuid4()), 'created_at': int(time.time()), 'data': data}

    return app
