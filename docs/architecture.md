# Architecture

vTSFM is an external vLLM plugin starter. Its model and IO adapters inherit the
pinned community Chronos-2 integration. The project owns the registration and
extension points; inference still delegates to upstream behavior.

```mermaid
flowchart LR
    C[Numeric HTTP client] --> V[vLLM pooling endpoint]
    C --> N[Native HTTP comparator]
    V --> I[vTSFM IO adapter]
    I --> M[vTSFM model adapter]
    M --> U[Pinned Chronos-2 implementation]
    N --> P[Official Chronos pipeline]
```

The native and vLLM servers run sequentially on the same GPU during a comparison.
They are alternatives in this diagram, not simultaneous resident services.

## Code boundaries

| File / directory | Responsibility |
| --- | --- |
| `src/vtsfm/config.py` | Pure constants, environment pins, model revision, default port |
| `src/vtsfm/plugin.py` | Lazy model registration and IO factory for the `vtsfm` entry points |
| `src/vtsfm/models/chronos2.py` | Thin upstream model subclass and explicit multimodal processor registration |
| `src/vtsfm/io_processor.py` | Thin upstream numerical IO subclass |
| `src/vtsfm/cli.py` | Build commands/environments and start the selected backend without GPU imports for help |
| `src/vtsfm/native_server.py` | Official Chronos pipeline behind the comparable HTTP boundary |
| `src/vtsfm/benchmark_memory.py` | Optional worker memory counters used in benchmark mode |
| `benchmarks/` | Data preparation, payloads, process ownership, metrics, raw results, plots |
| `scripts/` | Per-checkout setup, environment checks, serving, requests, smoke runs |
| `tests/` | CPU behavior checks and opt-in GPU integration checks |

## Plugin selection

The project registers `vtsfm` in the model plugin and IO processor entry-point
groups. The pinned Chronos package also registers `chronos2`. Serving must select
the intended entry point explicitly, so discovery order does not overwrite the
project's adapter with the community comparator.

The project model subclass has its own multimodal processor registration. This
registration is attached to the concrete model class and cannot be assumed to
follow ordinary Python subclass inheritance. Keep registration lazy so a CPU
checkout can import the package and inspect commands without importing vLLM or
Torch.

The serving command uses the pinned model revision, pooling runner, numerical IO,
FP32, and eager execution. Development RPC and memory hooks are enabled only in
benchmark mode, not normal serving. Bind to loopback unless access from another
machine is deliberately configured.

## Benchmark contract

Each comparison uses native HTTP and one plugin (`vtsfm` or `chronos2`). The runner
starts and stops only its own children, refuses occupied ports and GPUs, and
does not terminate existing services. Backend loading and warmup happen outside
the timed interval. Cancellation must release owned children and preserve partial
results for inspection.

Requests use matching numerical inputs, prediction length 96, quantiles 0.1/0.5/0.9,
and `cross_learning=false`. B is the number of sensor/history windows in one
request. The reference experiment has one outstanding HTTP request, so it
measures request batching rather than concurrent-client scalability.

End-to-end latency covers submitting a pre-encoded request through receiving and
decoding its response. Summary p95 pools the raw samples rather than averaging
block percentiles. Requests/s is the reciprocal of mean latency in seconds;
series/s is B divided by mean latency. With 150 samples per point, p95 depends on
a small tail and should not support strong claims about small differences.

GPU memory includes loaded/warmed service footprint and an NVML sampled peak.
The reference sampler interval is 20 ms, so it can miss short transients. Torch
allocated/reserved counters describe allocator state and are reported separately
from whole-service memory. Native in-process timing is a supplementary diagnostic,
not a symmetric model-only speedup measurement against vLLM.

### Input data

The default dataset is the [AssetOpsBench MetroPT fixture](https://github.com/IBM/AssetOpsBench/blob/e844f6468c6c68b328022e7ca8e31da140842cc7/src/couchdb/scenarios_data/shared/iot/metro_pump_1.json)
at commit `e844f6468c6c68b328022e7ca8e31da140842cc7`. The raw file's SHA256 is
`363d6daf873c614fcb4a798d4e86aa648d60ae2b6c735ce5d8388b0d4b3c2fdf`.
The AssetOpsBench paper identifies [MetroPT-3](https://doi.org/10.24432/C5VW3R)
compressor data as an upstream source. The exact raw-to-hourly aggregation recipe
is not documented in the inspected repository; the preparer performs no new
aggregation.

Seven analog channels share a 5,116-position hourly grid. All seven values are
missing in 560 rows. Preparation causally forward-fills these rows from preceding
observations, never future values, then casts to FP32. It preserves every observed
numeric value before that cast and records source checksums, timestamps,
per-channel metadata, and imputed indices in a provenance manifest.

The supported reference contexts are 128, 512, 2,048, and 4,096, with a 96-position
holdout. Windows from one compressor and overlapping history are correlated; they
are not independent assets. The experiment supports inference performance
comparisons and numerical backend parity checks, not forecasting-accuracy claims.
Synthetic numerical inputs are permitted only for explicitly labeled smoke tests.

## Future research boundaries

Cross-request packing, grouped attention kernels, TimesFM support, and MCP
integration are not implemented in this starter. Keep proposed changes behind
clear boundaries and demonstrate numerical parity before interpreting performance.
Any dependency, precision, scheduler, or IO change needs a separately labeled
baseline so results from different configurations do not become interchangeable.
