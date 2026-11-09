# Request lifecycle and offline forecasts

TimesFM, Toto, and Moirai use stateless shared IO on **vLLM 0.33.0 Model Runner
V1, FP32, eager execution, one GPU**. Chronos-2 retains its separate upstream
integration. Real-checkpoint lifecycle validation currently targets TimesFM.

## State ownership

Each preprocessing invocation receives a fresh numerical ID, even if its
external HTTP `X-Request-Id` repeats or its offline ID is absent. This ID travels
with the context through multimodal transport and worker-slot routing. The HTTP
response ID and schema remain vLLM's own.

The frontend IO processor has **no pending metadata map**. A completed pooler
result contains everything needed to format that request's forecast:

| Part | Contents |
| --- | --- |
| 28-byte header | Magic, format version, metadata length, series count, native quantile count, horizon |
| UTF-8 JSON | Item IDs, horizon, requested quantile levels; at most 1 MiB, checked before submission |
| Alignment | Zero padding to a four-byte boundary |
| FP32 bytes | Native forecast `[series, native_quantiles, horizon]`, without precision conversion |

The pooler returns one contiguous uint8 vector per scheduled request. Small
metadata moves to the forecast device; packing copies forecast bytes into that
vector, making a strided crop contiguous when needed. vLLM performs the normal
result copy to the CPU. Postprocessing validates the header and exact length,
views the FP32 payload without another copy, and formats the existing JSON
response. This is an internal plugin format, not a new public HTTP schema.

The worker pooler retains only bounded per-forward CPU layout/metadata until
pooling consumes it once. Each forward resets it before executing. Neither IO
nor pooler stores a collection of outstanding GPU tensors. Cancelling or failing
before postprocessing therefore leaves no frontend entry to remove.

vLLM owns queued/running requests and multimodal caches. Its HTTP handler and
`AsyncLLM` abort requests on disconnect/cancellation, and remove output
collectors. Cancelling an in-flight whole-forecast request cannot preempt its
CUDA work: that forward can finish before the abort is applied. Other requests
retain their own results. A caller using `AsyncLLM` directly must follow vLLM's
abort/iterator-lifecycle contract.

Validation, submission, and postprocessing failures are recoverable request
errors. A model exception inside V1's worker can terminate EngineCore; the
service then requires a restart. The plugin does not catch such exceptions and
pretend that a failed engine remains healthy. HTTP client timeouts are tested as
disconnects; the plugin does not add a server-side deadline policy.

## Offline input

Configure `LLM` with the same adapter, checkpoint revision, architecture override,
IO plugin, worker class, and V1/eager settings as the serving CLI. The public
plugin call is:

```python
payload = {
    'inputs': [
        {'item_id': 'pump-a', 'target': [1.0, 2.0, 3.0]},
        {'item_id': 'pump-b', 'target': [4.0, 3.0, 2.0]},
    ],
    'parameters': {'prediction_length': 96, 'quantile_levels': [0.1, 0.5, 0.9]},
}
result = llm.encode({'data': payload}, pooling_task='plugin')
predictions = result[0].outputs['predictions']
```

Use one dictionary `DataPrompt` per call; multiple series belong in its `inputs`
array. The pinned offline plugin dispatch does not support a list of dictionary
DataPrompts. Sequential calls may reuse the same `LLM` instance and omit IDs.
Thread-concurrent calls to the synchronous `LLM` engine are unsupported; use the
HTTP/async serving path for concurrent requests. Stateless IO does not make the
synchronous upstream engine thread-safe. HTTP wraps the same predictions with
its `request_id` and `created_at` fields; offline returns a list containing one
`PoolingRequestOutput`, whose `outputs` is the plugin dictionary.

## Manual checks

On an idle Linux GPU, using the checkout's configured TimesFM environment:

```bash
CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_ROUTING_ENGINE=1 PYTHONPATH=src:tests \
  .venv-timesfm25/bin/python -m pytest \
  tests/test_forecast_lifecycle.py tests/test_result_transport.py -q -s

CUDA_VISIBLE_DEVICES=0 VTSFM_RUN_TIMESFM_LIFECYCLE=1 PYTHONPATH=src:tests \
  .venv-timesfm25/bin/python -m pytest tests/test_timesfm_lifecycle.py -q -s
```

The first command checks exact byte transport, reordered postprocessing, repeated
external IDs, abandoned preprocessing, and metadata bounds. The real-model
suite exercises public offline and HTTP entry points, request errors,
cancellation and engine failure/restart. Its fault injection, state endpoints,
and worker gates are test-only and explicitly imported by the test launcher;
normal serving has none of them. These checks are separate from Modal's bounded
CUDA contract job and from native model parity/accuracy/performance evaluation.
