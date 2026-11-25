# TimesFM 2.5 compilation and KV cache writes

Issue #72 compares the shipped TimesFM guarded compile scope with wider Inductor scopes. The adapter also has a paired experiment that removes the CUDA scalar read in the pinned TimesFM cache writes, so the experiment can measure that change separately from compilation. All candidates use the existing vLLM whole-forecast path and the pinned official TimesFM checkpoint. Eager remains the default.

| Arm | Forecast implementation | Inductor scope |
| --- | --- | --- |
| Eager | Existing TimesFM adapter | None |
| `core` | Existing TimesFM adapter | Neural forward with native RMSNorm, activations and SDPA calls and reduced Inductor fusion |
| `wide` | Existing TimesFM adapter | Neural forward with default Inductor fusion |
| `whole` | Existing TimesFM adapter | Numerical forecast including outer normalization, recurrence and quantile assembly, with default Inductor fusion |
| `positioned-eager` | Adapter-owned dense KV cache writes at CPU-known positions | None |
| `positioned` | Same adapter-owned dense KV cache writes | Neural prefill with default Inductor fusion |

All compile arms keep Triton contraction and Inductor CUDA graphs disabled. The native adapter's attention reads a tensor-valued cache position to form a Python slice. A Dynamo trace for the wide arm broke at this upstream write. The positioned arms reuse the source-checked block implementation for static prefill and continuation writes. Both arms run the same eager forecast orchestration and continue with eager neural calls after prefill, avoiding a separate compile specialization for each continuation position. Each call owns its KV tensors and output. The model retains native logical positions, masks, running normalization, flip invariance and all nine quantile outputs.

## Matched A100 HTTP results

At `f0be810`, eager, core, wide and whole ran in that order and then in reverse on an A100-SXM4-40GB with FP32, TF32 off, vLLM 0.33.0 and Torch 2.14.0+cu130. The two orders also reversed input shape order. Each mode completed 200 requests per workload and client count across two fresh servers, for 3,200 requests overall. One numerical server slot served each request's own series. The four-client cells include queueing behind that slot. [Per-cell timing, cold costs, parity and memory](http-summary.json)

| MetroPT geometry | Clients | Eager req/s · p95 ms | Core req/s · p95 ms | Wide req/s · p95 ms | Whole req/s · p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1/C2048/H96 | 1 | 10.76 · 96.99 | 10.49 · 99.97 | 13.32 · 77.58 | 29.37 · 36.34 |
| B1/C2048/H96 | 4 | 11.60 · 359.35 | 11.16 · 378.08 | 14.44 · 295.51 | 34.18 · 130.23 |
| B4/C512/H96 | 1 | 15.56 · 69.08 | 15.19 · 70.28 | 21.89 · 47.97 | 28.24 · 37.59 |
| B4/C512/H96 | 4 | 17.91 · 226.65 | 17.16 · 248.27 | 26.21 · 158.27 | 35.93 · 115.72 |

Whole compilation gave 1.81–2.95× warm throughput over eager on these repeated shapes. First forecast specialization with fresh per-worker caches took 93–98 s for B1/C2048 and 41–46 s for B4/C512. Peak Torch allocated memory across cells was 933 MiB for eager and 954 MiB for whole. Peak sampled NVML residency was 2,045 and 2,111 MiB respectively. Timings exclude first use, warmups and parity queries. The independent native checks before and after timing passed all quantiles, with maximum absolute difference 0.00000763. The chiller case below evaluates a different workload.

## Numerical qualification

The wide and whole scopes matched upstream in the retained FP32 B1/C512/H96 and B16/C2048/H800 regression, including changed values, repeat calls after shape changes and input immutability. The wide test observed 24 unique Dynamo graphs and the whole test observed 42.

The separate five-signal AssetOpsBench HTTP check passed four signals in every arm. Chiller condenser flow produced strict parity misses in the same input row across ordinary eager, wide and whole. Each case had 4,320 quantile values. All three native reference forecasts matched exactly. The configured tolerance was 0.0001 absolute plus 0.0001 relative.

| Scope | Chiller misses | Shared with eager | Candidate-only misses | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| Eager | 69 | 69 | 0 | 0.00390625 |
| Wide | 57 | 33 | 24 | 0.00610352 |
| Whole | 59 | 46 | 13 | 0.00366211 |

The wider scopes reduced the total miss count on this input and introduced new failures at other quantile cells. Broad compilation remains opt-in.

## Position-known prefill ablation

The two A100 subprocesses at `f9f9e51` used the same independently generated native forecast. Both passed the short, changed recursive and ragged synthetic cases and four of five AssetOpsBench signals. Chiller condenser flow missed tolerance for **69 of 4,320** quantile values per invocation in the position-known eager control and **50 of 4,320** per invocation with compiled prefill. The largest absolute difference on this signal was 0.00390625 and 0.00366211, respectively. Each arm compared 747,072 values, including repeat invocations.

| Direct workload | Positioned eager warm call (ms) | Positioned compile warm call (ms) | First compiled call (s) |
| --- | ---: | ---: | ---: |
| B1/C512/H96, two visits | 47.87–50.94 | 19.99–24.89 | 51.44 |
| B16/C2048/H800, three visits | 333.09–341.40 | 319.35–342.89 | 39.13 |
| Ragged two-row H256, one visit | 80.96 | 53.48 | 38.93 |
| B5/C512/H96, five signals | 47.66–51.49 | 22.78–33.63 | 47.96 |

Each visit has one retained warm sample. Four shapes each captured one prefill graph. The H96 rows show where compiling prefill reduces the cost of a direct forecast. In the H800 test, the ranges overlap because the forecast spends more time in eager continuation. This pair has no measured HTTP speedup, and the chiller result leaves both paths experimental.

The shared-checkout archive at `results/issue72-timesfm-agent-20261125/positioned-direct.tgz` includes both case manifests, identical native arrays, per-case graph counts and parity distributions, sampled ownership and cleanup. SHA-256 `1e5029188a9ae49f046f1b430749f0a46cc3bc6339df4cc1c4452612a8ee384c`. The two candidate subprocesses recorded 757 and 41 ownership checks, with zero foreign processes in either run. The parent checked process ownership approximately every 250 ms. The recorded monitor method also mentions an HTTP memory sampler; direct timing used no sampler.

## Flip batching and strict native parity

The pinned official pipeline decodes the original and sign-flipped batch in two calls. vTSFM normally concatenates them to reduce kernel launches. A direct FP32 A100 experiment compared both decode layouts under the official and vectorized quantile assembly. Both packed arms missed the same 69 of 4,320 condenser quantiles. Both separate-pass arms matched all 4,320 values bit for bit. The difference was already present in the decoder streams before quantile assembly. Warm direct-call medians were about 53 ms packed and 103 ms split over five calls per cell.

The actual split eager HTTP worker at `c2af55b` matched all 21,600 values bit for bit on the five B5/C512/H96 signal cases. It also matched all 3,690 values across six integration cases, including ragged inputs, sparse quantile selection, repeated inputs after shape changes and a recursive forecast. Guarded-core compilation with split flips missed 33 and whole compilation with split flips missed 41 of 4,320 condenser values. All other signals passed their strict tolerance. The measured 1.81–2.95× warm HTTP results above use packed flips.

Set `VTSFM_TIMESFM_FLIP_LAYOUT=split` to use the separate-pass option for independent whole forecasts in FP32. This selection rejects cross-request fusion, block scheduling, and fixed-shape graph replay because they use different batch or cache geometry. Packed eager remains the default. The direct diagnosis is reproducible with `python -m benchmarks.timesfm_flip_ablation --archived-native <native.json> --output <new.json>` using the independently saved TimesFM five-signal native file.

## CPU checks

The isolated branch passed 784 CPU tests in the standard workspace environment, with 85 dependency skips and 208 GPU cases deselected. The pinned Linux Torch/TimesFM environment passed 54 targeted tests with CUDA hidden. Its two failures both came from the pre-existing shared-prefill storage test, which expects separate layer storages. Both reproduce on the unchanged `2e13348` baseline, where each owner already packs disjoint layer views into shared storages. Four tests for other adapters skipped in the pinned TimesFM environment.

## Reproduction

Use a fresh process per scope in the pinned Linux TimesFM environment on an idle GPU. To compare immediate first-use cost and warm synchronized tensor calls, run the direct gate against independently loaded official TimesFM predictions. It stores every native and served comparison summary, even where parity fails. Warm readings are a mechanism diagnostic. Use the separate matched HTTP comparison for serving claims.

```bash
py=.venv-timesfm25/bin/python
export PYTHONPATH=src:tests:.
"$py" -m benchmarks.prepare_data --assetops-signals --output data/model-integration

for scope in eager core wide whole positioned-eager positioned; do
    "$py" -m benchmarks.timesfm_compile_gate --scope "$scope" --retained \
        --data-dir data/model-integration --warm 1 --output "results/timesfm-compile-$scope"
done
```

The synthetic retained sequence exercises B1/C512/H96, B16/C2048/H800, changed B16 values, ragged histories at H256 and return to the first geometry. The five AssetOpsBench cases each use five rows with C512/H96. The gate checks every row and all nine quantiles with `atol=rtol=1e-4` and records misses by row, quantile and 128-step block. First-use timings measure the synchronized tensor call and exclude model load, input staging, oracle calculation and parity checks. The first call for a case can reuse a graph from an earlier matching geometry, so use the graph count and case order when interpreting cold costs. A native parity failure exits with status 3 after recording the completed cases.

The corresponding serving setup uses `VTSFM_TIMESFM_COMPILE_SCOPE` and `--forecast-execution compile`. Select eager for the `positioned-eager` control. Experimental scopes require FP32 and whole-forecast serving.

For the serving comparison, start each arm with a fresh output directory and reverse both the arm and shape orders for the second block:

```bash
"$py" -m benchmarks.toto_compile --adapter timesfm25 --arm whole \
    --data-dir data/model-integration --output results/timesfm-whole-first \
    --shapes 1,2048,96 4,512,96 --clients 1 4 --requests 100
```
