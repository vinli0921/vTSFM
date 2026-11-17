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

## Compare TimesFM flip batching (issue #52)

Use two clean checkouts on the same idle GPU. Run these commands for each
revision, with the same pinned data, model, FP32 settings, and environment.
Set `RUN_LABEL` to a unique value for each pass and repeat the
untraced comparison in baseline, candidate, candidate, baseline order.

```bash
# From the root of each checkout. Set NSYS, AOB_ROOT and HF_HOME as above.
export PYTHONPATH="$PWD/src"
py=.venv-timesfm25/bin/python
out="results/timesfm-flip-$(git rev-parse --short HEAD)-${RUN_LABEL:?set a unique run label}"
mkdir -p "$out"
git rev-parse HEAD > "$out/revision.txt"

# The runtime path bypasses HTTP admission. 0b runs B64/C2048/H96.
for round in 1 0b16 0b; do
  "$py" benchmarks/nsight/driver.py --model timesfm25 --path runtime \
    --round "$round" --warm 10 --n 100 --out "$out/runtime-${round}.json"
done

# Trace separately so profiling overhead stays out of the latency comparison.
"$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$out/runtime-b1" "$py" benchmarks/nsight/driver.py --model timesfm25 \
  --path runtime --round 1 --n 50 --out "$out/runtime-b1-traced.json"
"$NSYS" export --type sqlite -o "$out/runtime-b1.sqlite" "$out/runtime-b1.nsys-rep"
"$py" benchmarks/nsight/analyze.py "$out/runtime-b1.sqlite" > "$out/runtime-b1-analysis.jsonl"
```

Use the untraced JSON files for mean, p50 and p95 and the trace for kernels and
launch calls per request. For matched HTTP timings and traces, use
`benchmarks/nsight/run.sh <output> timesfm25` on each revision. It uses B16 for
the batched HTTP round. Report any B64 runtime OOM or slowdown as a measured
result. Recheck peak memory for the admitted whole and block workloads as
described in [numerical admission](../../docs/numerical-admission.md).
