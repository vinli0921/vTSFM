# Third-party software and data

vTSFM project code is licensed under Apache-2.0. Dependencies and downloaded data
retain their own licenses; the project license does not replace those terms.

## Chronos forecasting integration and model

- Project: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- Integration: [PR #473](https://github.com/amazon-science/chronos-forecasting/pull/473),
  commit `ba47d25a04aedac33af2fb9780609afd54744d5e`.
- License: [Apache-2.0](https://github.com/amazon-science/chronos-forecasting/blob/ba47d25a04aedac33af2fb9780609afd54744d5e/LICENSE).
- The vTSFM model/IO classes extend this integration and delegate inference to it.
  The upstream package is installed as a dependency rather than vendored here.
- Model: [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2), revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c`; retain its model-card attribution
  and license when downloading or redistributing model artifacts.

## AssetOpsBench fixture

- Project: [IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench).
- Pinned commit: `e844f6468c6c68b328022e7ca8e31da140842cc7`.
- Source path: `src/couchdb/scenarios_data/shared/iot/metro_pump_1.json`.
- Repository license: [Apache-2.0](https://github.com/IBM/AssetOpsBench/blob/e844f6468c6c68b328022e7ca8e31da140842cc7/LICENSE).
- Attribution: Patel et al., [AssetOpsBench](https://arxiv.org/abs/2506.03828).
- The preparer downloads this fixture on request, selects seven analog sensor
  columns, causally forward-fills wholly missing rows, and converts to FP32.
  It records the exact source checksum and transformation in the data manifest.
  Raw sensor data is not included in this repository.

## MetroPT-3 upstream data

- Dataset: Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021).
  *MetroPT-3 Dataset*. UCI Machine Learning Repository.
  [DOI 10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R).
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
  as stated by the [UCI dataset page](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B).
- The AssetOpsBench paper identifies MetroPT-3 as a source of compressor data.
  Its raw-to-hourly fixture aggregation recipe was not documented in the inspected
  repository. Our further changes are limited to column selection, documented
  causal forward-fill, and FP32 conversion.
- Preserve this citation, source links, license link, and description of changes
  when sharing derived fixture data. Do not describe imputed values as observed.

## Other runtime dependencies

vLLM, PyTorch, Transformers, NumPy, FastAPI, and other installed packages ship their
own license information. Consult the installed distributions and upstream license
files when redistributing an environment. GPU drivers and CUDA components have
separate vendor terms and are not included here.
