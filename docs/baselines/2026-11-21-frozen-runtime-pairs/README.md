# Repeating the vLLM 0.16 to 0.33 upgrade on the A100

Toto was the only model that served every workload faster on the newer stack in all three trials. Across all models, throughput increased **1.1%** and p95 latency decreased **1.8%**. The original spike had found a small overall regression. The three-trial result applies to these two frozen releases and this AssetOpsBench workload. [Issue #33](https://github.com/vinli0921/vTSFM/issues/33) · [Original spike](../../runtime-upgrade.md)

## Results across the three trials

Each cell gives the percentage change in requests/s followed by the percentage change in p95 HTTP latency when moving to 0.33. The geometric mean gives equal weight to each workload and trial. The memory column gives the largest sampled whole-device footprint in each arm.

| Model | Original spike | Trial 1 | Trial 2 | Trial 3 | Three trials | Peak NVML GiB, control / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimesFM 2.5 | −4.5 / +3.5 | +0.5 / -1.0 | +1.4 / -3.7 | -0.8 / +0.1 | +0.4 / -1.5 | 1.98 / 1.98 |
| Toto 2 | +4.5 / −4.4 | +2.6 / -3.3 | +5.0 / -5.8 | +3.5 / -3.4 | +3.7 / -4.2 | 2.39 / 2.40 |
| Moirai 2.0 | −4.8 / +4.5 | -0.8 / +0.4 | +0.4 / -0.7 | -1.3 / +1.0 | -0.5 / +0.3 | 1.14 / 1.14 |
| All models | −1.7 / +1.1 | +0.8 / -1.3 | +2.3 / -3.4 | +0.4 / -0.8 | +1.1 / -1.8 | 2.39 / 2.40 |

Trials 1 and 3 ran 0.16 first. Trial 2 reversed both the stack order and the workload order within each stack. The per-trial columns show how much the estimate varied. With only one reverse-order trial, the effect of order alone cannot be estimated.

## Workloads for profiling

Across the 24 workloads, 11 showed higher throughput and lower p95 in **all three** trials, one showed the opposite, and 12 changed direction on at least one metric. The [per-workload table](workload-variation.csv) gives each trial’s change, range and direction. The [72 paired measurements](paired-trials.csv) include both arms’ absolute requests/s, series/s, p50 and p95, sampled peak bytes, payload hash and parity error.

- **Toto speedup.** All eight Toto workloads improved on both metrics in every trial. With 128 observations, four series per request and four concurrent clients, throughput rose **3.8–5.4%** and p95 fell **4.7–8.4%**. This is the first workload to profile in [issue #34](https://github.com/vinli0921/vTSFM/issues/34).
- **Moirai slowdown.** The single repeatable slowdown was the original outlier, with 512 observations, one series and four clients. Throughput fell **0.5–1.7%** and p95 rose **1.4–2.9%**. The original spike had measured a 16.0% p95 increase with 100 requests and 2.1% on its 200-request repeat. Each arm below timed 200 requests.
- **Variable direction.** Seven of eight TimesFM workloads and five of eight Moirai workloads changed direction on at least one metric. For example, TimesFM with 512 observations, four series and four clients ranged from **−1.3% to +1.7%** in throughput and **−6.8% to +0.8%** in p95. Comparing a single pair at this point can give the wrong direction.

| Pair | First stack | p95 ms 0.16 | p95 ms 0.33 | p95 change | Requests/s change |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 0.16 | 146.9 | 151.2 | +2.9% | -1.7% |
| 2 | 0.33 | 147.7 | 150.2 | +1.7% | -0.5% |
| 3 | 0.16 | 148.5 | 150.5 | +1.4% | -1.4% |

## How the comparison was run

| | Control | Candidate |
| --- | --- | --- |
| vTSFM revision | [65bd0c8](https://github.com/vinli0921/vTSFM/commit/65bd0c8171754c920da36094439fad8f6cc6a490) | [f122dda](https://github.com/vinli0921/vTSFM/commit/f122ddab88cdd26ece4446b18ddcf4fcbb137d28) |
| vLLM · Torch · CUDA runtime · Transformers | 0.16.0 · 2.9.1 · 12.8 · 4.57.6 | 0.33.0 · 2.14.0 · 13.0 · 5.20.2 |

Both revisions ran in their original per-model environments with clean source trees. Their checkpoints were identical across stacks: TimesFM `1d95242`, Toto `a7bab28`, and Moirai `30f43ff`. The A100-SXM4-40GB and driver 595.91.07 matched the original hardware configuration. The device UUID differed from the November 5 spike, and stayed constant across the new trials.

The client replayed the original AssetOpsBench MetroPT compressor HTTP payloads. The request-body SHA-256 matched the archived spike for every workload in both arms. Each request forecast 96 future steps and nine quantiles for one or four series with 128 or 512 context steps. One or four concurrent clients sent requests. Both servers used the V1 pooling runner, eager FP32, TF32 disabled, one scheduler slot, and a 0.35 GPU memory fraction.

Each trial launched a new server for each model and release. After five warmups per point, the client timed 100 requests, or 200 for Moirai with 512 observations, one series and four clients. Latency covers sending the pre-encoded HTTP body through decoding the response. Throughput is completions divided by elapsed wall time. The latency percentiles use the individual request samples. NVML sampled whole-device resident memory approximately every 20 ms and can miss shorter peaks. The client checked forecast values separately from timing.

The runner waited for an idle GPU, checked GPU process ownership during measurement, and recorded teardown after each arm. An interruption disqualified both sides of that model pair. Its files were kept separately and a fresh pair was run.

## Verification and evidence

All 15,000 timed requests succeeded across the 18 accepted server launches. In the separate numerical check, all 155,520 cross-stack forecast values met the original absolute and relative tolerances of `1e-4`. The maximum absolute difference was `3.8147e-6`. Six interrupted attempts were excluded. GPU ownership checks passed for the accepted runs and the device was idle after final cleanup.

The team copy of the evidence is at `hpml:~/vtsfm-issue33-20261121/frozen-runtime-pairs-A100-20261121.tar.gz`. Its SHA-256 is `cc716eb5374974f7cbcd13739a3ef41ed0aa4a7a716a5c2937414cf5710d4683`. The archive includes individual HTTP latencies, NVML memory samples, request hashes, forecast arrays, full environment manifests, the fixture and its provenance, and the excluded attempts. The bundled original archive supplies the historical payload hashes.

From the extracted directory, regenerate the summary and independently verify the raw samples with Python 3.11 or newer.

```bash
python3 analysis/summarize.py trials summary
python3 analysis/check_raw.py trials analysis/original-spike.tar.gz
```
