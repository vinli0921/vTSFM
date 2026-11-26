# Moirai 2 forecast-block serving on the A100

Whole-forecast serving had higher throughput and lower p95 latency on both AssetOpsBench workloads. The experimental scheduler interleaved requests every 64 forecast points and lowered peak GPU allocation. Keep whole-forecast serving as the default for these workloads.

| Workload | Mode | Requests/s | Offered p95 (s) | Peak Torch allocated (MiB) | Peak NVML residency (MiB) |
|---|---|---:|---:|---:|---:|
| H1024 saturation | Whole | 11.33 | 8.36 | 179.79 | 1,372.56 |
| H1024 saturation | Stepped | 3.81 | 25.30 | 85.36 | 1,158.56 |
| Mixed burst | Whole | 8.29 | 10.22 | 115.20 | 1,372.56 |
| Mixed burst | Stepped | 7.86 | 10.72 | 85.36 | 1,158.56 |

Both server orders ran on an A100-SXM4-40GB with vLLM 0.33.0, Torch 2.14.0+cu130, uni2ts 2.0.0, four slots, eight active rows and eager FP32 with TF32 disabled. The measured implementation was `bbf77ea2d2779d36eb1b7f8d1bb24e947a293a0e`. Saturation offered two 512-observation histories with H1024 every millisecond. Mixed bursts offered 16 requests every 250 ms with horizons 128, 129, 513 and 1024. The saved summary includes exact shapes, arrival order, input hashes, pin, individual trial results, per-class p95 and allocator counters. [Machine-readable results](summary.json)

All 800 timed requests succeeded. Each table cell pools 200 requests across the two reversed server orders. Throughput divides completions by summed elapsed time. Latency starts at the scheduled offer and ends after HTTP decoding. Torch tracks worker allocator peaks. NVML sampled device residency every 20 ms. No foreign GPU process appeared in any accepted stage and each stage released its GPU processes. The initial contended attempt was excluded.

## Why the prototype isolates owners

On a synthetic H1024 counterexample, one request supplied 79- and 81-point histories and a second supplied another 81-point history. Native `predict` batches by true history length *within each request*. Holding batch geometry fixed, whole and stepped execution produced identical outputs for all three rows. Merging owners into one model call changed some quantiles by up to 0.02838 relative to independent native inference. Merging just the two equal-width rows still produced a maximum difference of 0.02195. The prototype therefore preserves the native length groups and interleaves requests at round boundaries. This counterexample also reproduced on the ordinary whole wrapper for these synthetic inputs.

The real checkpoint passed 10 direct CUDA checks. The V1 HTTP lifecycle test matched all 258,761 returned values exactly, covered cancellation, replacement and malformed input, and completed three eight-row C7168/H1024 waves. Live allocation returned to 54,461,440 bytes after each wave and peaked at 1,952,331,776 bytes. In the controlled arrival trace a short request completed 682 ms before the first long request. All 160 concurrent warmup responses in the timed workload configurations passed independent CUDA native checks at `atol=rtol=1e-4`. The timed requests checked HTTP status, identity and series count. These replays measure serving on repeated compressor histories; future-label forecast accuracy was outside this comparison.

## Reproduce on an idle Linux A100

From the repository root, reserve the GPU with the other users of the node. Use the pinned model environment and choose a new output directory for each run.

```bash
./scripts/setup.sh --adapter moirai2
py=.venv-moirai2/bin/python
fixture=data/model-integration
out=results/moirai-forecast-blocks
qual=docs/baselines/2026-11-25-moirai-forecast-blocks/qualify_replay.py
mkdir -p "$out"
"$py" -m benchmarks.prepare_data --output "$fixture"

export PYTHONPATH="$PWD/src:$PWD/tests:$PWD" CUDA_VISIBLE_DEVICES=0
export VTSFM_EXPERIMENTAL_MOIRAI_BLOCKS=1 NVIDIA_TF32_OVERRIDE=0
export VTSFM_PRECISION=fp32 VTSFM_TEST_PRECISION=fp32 VTSFM_FORECAST_EXECUTION=eager
export VTSFM_MOIRAI_ATTENTION=upstream OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
VTSFM_TEST_ADAPTER=moirai2 VTSFM_MOIRAI_BLOCK_CHECKPOINT=cuda \
  "$py" -m pytest -q tests/test_moirai_stepping.py
VTSFM_RUN_MOIRAI_CONTINUOUS=1 \
  "$py" -m pytest -q tests/test_moirai_continuous.py
"$py" "$qual" oracle "$fixture" "$out/native.json"
"$py" "$qual" replay "$fixture" "$out/native.json" "$out/ab" --modes whole,continuous
"$py" "$qual" replay "$fixture" "$out/native.json" "$out/ba" --modes continuous,whole
```

Each replay writes raw samples, separate concurrent native checks and a benchmark summary. The team archive contains the exact original scripts, GPU process ownership records, matched arithmetic controls, failed fusion evidence and accepted HTTP samples. It is available at `hpml:~/vtsfm-final-20261125/blocks/moirai-forecast-blocks-A100-20261125.tar.gz`, SHA-256 `e134e6d3ad0b94612143c73af436d96ee2895340d02c41f7ca96943a9085b6a7`.
