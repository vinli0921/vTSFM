# vLLM 0.33 runtime upgrade

TimesFM 2.5, Toto 2, and Moirai 2.0 use the
[published vLLM 0.33.0 wheel](https://pypi.org/project/vllm/0.33.0/) with
PyTorch 2.14.0+cu130 and Transformers 5.20.2. The launcher selects Model
Runner V1. Chronos-2 retains the vLLM 0.16.0 / Torch 2.9.1+cu128 stack used
by its pinned upstream plugin. Model revisions and the numerical HTTP
request/response schema stay the same.

## Validation

The control was the vLLM 0.16.0 V1 output-routing revision (`65bd0c8`). Both
stacks ran on the same A100-SXM4-40GB with driver 595.91.07, FP32 with TF32
disabled, and matching checkpoint revisions. The manual integration profile
checked 18 HTTP cases against native GPU outputs, including six from
AssetOpsBench sensor windows. It checks the returned item IDs, horizon,
quantiles, and values with absolute and relative tolerance `1e-4`.
The six checkpoint-free transport and routing checks passed. At three
scheduler slots, each real model also passed 12 mixed-shape HTTP requests
against its isolated responses. All five CUDA tensor contract checks passed.
The three CPU checkpoint smoke tests also passed.
On the unchanged CUDA 12.8 Chronos environment, both the vTSFM and upstream
backends passed an HTTP forecast smoke check through the candidate launcher.

To rerun on Linux with an idle GPU, use a checkout that includes the manual
model profiles from `main`:

```bash
for adapter in timesfm25 toto2 moirai2; do
  ./scripts/setup.sh --adapter "$adapter"
done
./scripts/test-models.sh --profile integration --gpu 0
```

For performance, compare the two vTSFM HTTP servers on the same prepared
AssetOpsBench compressor fixture. The matrix uses 1 or 4 series per request,
128 or 512 observations per series, a 96-step horizon, and 1 or 4 clients.
Each point times 100 requests after 5 warmups, from sending the pre-encoded
body to decoding the response. GPU memory is peak whole-device NVML
residency sampled every 20 ms; shorter peaks may be missed. Both release
arms used one scheduled request per forward and a GPU memory fraction of
`0.35`. Use
`scripts/test-models.sh --profile performance` for the same workload matrix
and measurements, including a native HTTP baseline on each stack.

| Adapter | Requests/s | p95 latency | Peak GPU memory, old → new |
| --- | ---: | ---: | ---: |
| TimesFM 2.5 | −4.5% | +3.5% | 1.98 → 1.98 GiB |
| Toto 2 | +4.5% | −4.4% | 2.39 → 2.40 GiB |
| Moirai 2.0 | −4.8% | +4.5% | 1.14 → 1.14 GiB |

Percentages are geometric means of paired ratios across eight points per
adapter; memory is the largest measured peak per adapter. Across all 24
pairs, throughput was 1.7% lower and p95 was 1.1% higher. Every matched
forecast agreed within the parity tolerances; the largest absolute difference
was `3.82e-6`. Against observed future labels on the same windows, the
largest changes in MASE and scaled quantile loss were `2.81e-6` and
`5.90e-7`, respectively. These windows check that the runtime preserves
accuracy on the fixture; they do not rank the models. The upgrade has mixed
performance effects and should be treated as a new benchmark baseline.
The comparison measures the whole runtime stack; it cannot isolate the effect
of any single dependency.

Moirai's largest p95 increase in the grid was 16.0% (145.5 to 168.9 ms:
one series, 512 observations, four clients). A 200-request repeat with the
stack order reversed measured 151.2 versus 154.4 ms (+2.1%). Both
measurements are retained because the tail varied between runs.

Forcing asynchronous scheduling on vLLM 0.33.0 reduced throughput by
1.4–4.4% and raised p95 by 2.9–6.3% across the four-client points per model.
Giving Toto three scheduler slots reduced throughput by 1.2% and raised p95
by 2.9%. The launcher keeps the upstream pooling scheduler default and
one request slot.

The [frozen-stack attribution study](baselines/2026-11-21-frozen-runtime-attribution/README.md) compares direct forecasts and HTTP serving, tests both V1 scheduler modes and isolates the response encoder. It includes request and CUDA traces for all three models.

## Why V1

Stock V2 in the published wheel rejects the `plugin` pooling task. The V1
compatibility port uses the stock wheel and keeps vLLM's numerical request
transport and forecast response envelope. Each forecast still runs
separately. The upgrade does not add numerical continuous batching.

Moving to V2 would require carrying raw per-request numerical payloads and
request identities through its model state. The spike tried two ways to get
past the task restriction: a test-process allowlist change, and the stock
wheel's `embed&token_classify` task as an internal alias. Both passed mixed-shape
worker routing and checkpoint-free HTTP tests. The plain `embed` alias
required a tokenizer during offline startup, so it was unsuitable for these
numerical models.

The combined-task prototype also passed 12 concurrent requests with real
Toto weights. With four clients and three scheduler slots on both runners,
it was 1.2% higher in throughput and 3.5% lower in p95 across four matched
points than the V1 runner, with identical forecast values and the same
2.40 GiB peak.

Neither experiment ships. An allowlist override depends on a private constant,
and an alias relies on unrelated task semantics. See
[vLLM PR #56805](https://github.com/vllm-project/vllm/pull/56805) for a related
effort to support the native `plugin` task on V2.
