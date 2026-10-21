# Historical native / community Chronos baseline

[`summary.csv`](summary.csv) is a preserved summary from the 2026-10-20 reference
experiment. **Its `vllm` rows measure the community `chronos2` plugin, not the new
`vtsfm` plugin.** These are measurements before any vTSFM-specific optimization;
they establish historical context, not a vTSFM speedup claim.

The source artifact was the earlier experiment's
`vtsfm-benchmarks/2026-10-20/results-final/summary.csv`. This repository contains
only the small CSV summary, not raw sensor downloads, model weights, machine
logs, or a full raw measurement archive.

The copied CSV has 40 configuration/backend rows and SHA256
`95d8825c885e027d5658414e6180a00317f83dec867c4229a696842029317b38`.

| Setting | Historical value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB, one backend resident at a time |
| Python / Torch / CUDA | 3.11.17 / 2.9.1+cu128 / 12.8 |
| vLLM / Transformers | 0.16.0 / 4.57.6 |
| Chronos integration | `ba47d25a04aedac33af2fb9780609afd54744d5e` |
| Model revision | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| Precision / execution | FP32, eager, `OMP_NUM_THREADS=4` |
| vLLM scheduler | `max_num_seqs=1`, `max_num_batched_tokens=8192` |
| Grid | B = 1, 4, 16, 64, 128; C = 128, 512, 2048, 4096 |
| Forecast | H96, quantiles 0.1/0.5/0.9, `cross_learning=false` |
| HTTP sampling | Three blocks × 50 timed requests per point; ten warmups |
| Total timed HTTP samples | 6,000 across two backends and 20 configurations |
| Concurrency | One outstanding request, one client |
| Memory sampling | NVML every 20 ms; allocator counters reported separately |

Inputs came from the pinned AssetOpsBench hourly MetroPT fixture: seven channels,
5,116 hourly grid positions, and causal forward-fill of 560 wholly empty rows.
Observed values were retained before FP32 conversion. B identifies correlated,
potentially overlapping sensor/history windows from one compressor. These data
and measurements do not establish forecasting accuracy.

`mean_ms`, `p50_ms`, and pooled `p95_ms` describe complete HTTP request/response
latency, including response JSON decoding. `request_per_s` is 1/mean latency in
seconds; `series_per_s` is B/mean latency. At 150 samples per point, interpret
small p95 differences cautiously. `sampled_peak_gpu_mib` can miss transients
between sampler polls. Native in-process values are supplementary and have no
matched vLLM model-only timing counterpart.

The `loaded_gpu_mib` values describe a separately observed loaded baseline;
they are not guaranteed to be lower than the point's warmed or sampled peak.
Do not subtract these fields to infer an active working set. Payload hashes
allow matching rows across the two historical backends.

To produce current results, follow the repository's benchmark commands and label
the selected plugin explicitly. Re-run rather than treating this summary as a
measurement of a different adapter, dependency set, GPU, or concurrency level.
