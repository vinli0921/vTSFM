#!/bin/bash
# Build the vtsfm-prof env on the compute node's local disk (home is at quota).
# Run inside a Slurm allocation: srun --jobid=<id> --overlap bash ~/vtsfm/setup_env.sh
set -euo pipefail
ROOT=${VTSFM_ROOT:-/local/jc6980_vtsfm}
mkdir -p "$ROOT"
export UV_CACHE_DIR=$ROOT/uv-cache HF_HOME=$ROOT/hf
UV=${UV:-~/.local/bin/uv}
if [ ! -x "$ROOT/vtsfm-prof/bin/python" ]; then
  $UV venv -q --python 3.12 "$ROOT/vtsfm-prof"
fi
source "$ROOT/vtsfm-prof/bin/activate"
# versions pinned to the 2026-10-27 Insomnia baseline (results/env.txt) so reruns differ only
# in hardware; the Moirai-only deps (gluonts, lightning, ...) are new and left unpinned
$UV pip install -q --torch-backend cu128 "torch==2.11.0" \
  "sktime==1.2.0" "skpro==2.16.0" "chronos-forecasting==2.3.2" "toto-models==1.0.0" "toto-2==2.0.0" \
  "transformers==5.21.0" "numpy==2.5.4" "pandas==2.3.3" \
  accelerate huggingface_hub statsmodels scipy pyyaml nvtx \
  gluonts einops lightning hydra-core safetensors hf-xet
for m in amazon/chronos-2 google/timesfm-2.5-200m-transformers Datadog/Toto-2.0-313m \
         Salesforce/moirai-2.0-R-small; do
  hf download "$m" >/dev/null && echo "downloaded $m"
done
$UV pip list 2>/dev/null | grep -iE "^(torch|sktime|skpro|chronos-forecasting|toto-2|toto-models|transformers|timesfm|gluonts|lightning|numpy|pandas) "
