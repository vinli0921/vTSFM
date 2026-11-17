# Nsight Systems host-overhead profiling (issue #16)

Tools behind [docs/baselines/2026-11-10-nsight-host-overhead](../../docs/baselines/2026-11-10-nsight-host-overhead/README.md).
They split per-request wall time into GPU busy, CUDA launch, sync wait, copy API calls and
Python/framework time, for three paths per model:

| Path | What runs |
| --- | --- |
| `native` | Official upstream API: TimesFM `forecast()`, `Toto2Model.forecast`, uni2ts `Moirai2Forecast.forward` |
| `runtime` | The vTSFM adapter's `runtime.forecast`, called in-process as `ForecastModel.forward` does |
| `http` | `POST /pooling` to the vLLM server; one request outstanding (or the saturated block-mode case) |

| File | Role |
| --- | --- |
| `workload.py` | Harness data and windows (`profiling/harness.py` on `profiling-baseline`) |
| `driver.py` | In-process native/runtime runs; NVTX `request/preprocess/forward/postprocess`; timed requests bracketed by `cudaProfilerStart/Stop` |
| `launch_server.py` | Starts the server from `vtsfm.cli.build_vllm_command`, under `nsys launch` unless `--no-nsys` |
| `http_client.py` | Sends rounds `1`, `0b`, `0b16` or `sat`; calls `nsys start/stop` around the timed requests |
| `nvtx_hook/sitecustomize.py` | With `VTSFM_NVTX=1`, wraps about 40 vLLM/vTSFM request-path functions in NVTX ranges. Done at import time; no source edits |
| `analyze.py` | Per-request split from `nsys export --type sqlite` output |
| `run.sh` | All runs (each traced, plus an untraced timing run), then `nsys stats` CSVs and `analysis.jsonl` |

## Run

On a Linux GPU host with the adapter environments from `scripts/setup.sh --adapter <name>`:

```bash
export NSYS=/path/to/nsys AOB_ROOT=/path/to/AssetOpsBench HF_HOME=/path/to/hf
export METROPT_DATA=/path/to/prepared/assetops   # optional: TimesFM block-mode case
benchmarks/nsight/run.sh results/nsight-NEW timesfm25 toto2 moirai2
```

`nsys` can be installed without root by unpacking NVIDIA's `nsight-systems-*.deb` with
`dpkg-deb -x`. Only `chiller_6.json` is read from AssetOpsBench.

Notes:

- When `kernel.perf_event_paranoid` is above 2, nsys cannot sample CPU or Python stacks or record
  context switches. That is why the NVTX hook exists. Python/framework time is the remainder after
  CUDA API time on the compute thread.
- CUDA tracing adds roughly 6–8 µs per kernel launch. Compare traced splits against the untraced
  timing runs, not against absolute traced wall times.
- TimesFM admission caps a request at 16 rows, so its batched HTTP round is `0b16`.
