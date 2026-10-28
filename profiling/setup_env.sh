#!/bin/bash
# Build the vtsfm-prof env on the compute node's local disk (home is at quota).
# Run inside a Slurm allocation: srun --jobid=<id> --overlap bash ~/vtsfm/setup_env.sh
set -euo pipefail
ROOT=${VTSFM_ROOT:-/local/jc6980_vtsfm}
mkdir -p "$ROOT"
export UV_CACHE_DIR=$ROOT/uv-cache HF_HOME=$ROOT/hf
UV=~/.local/bin/uv
if [ ! -x "$ROOT/vtsfm-prof/bin/python" ]; then
  $UV venv -q --python 3.12 "$ROOT/vtsfm-prof"
fi
source "$ROOT/vtsfm-prof/bin/activate"
$UV pip install -q --torch-backend cu128 torch \
  "sktime==1.2.0" "skpro>=2.14" "chronos-forecasting>=2.0.0" toto-models \
  transformers accelerate pandas numpy huggingface_hub statsmodels scipy pyyaml nvtx
for m in amazon/chronos-2 google/timesfm-2.5-200m-transformers Datadog/Toto-2.0-313m; do
  hf download "$m" >/dev/null && echo "downloaded $m"
done
$UV pip list 2>/dev/null | grep -iE "^(torch|sktime|skpro|chronos-forecasting|toto-2|toto-models|transformers|timesfm|gluonts|numpy|pandas) "
