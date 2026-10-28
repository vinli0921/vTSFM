#!/bin/bash
# Run the profiling harness inside an existing Slurm allocation on an A6000 node.
#   srun --jobid=<id> --overlap bash profiling/run.sh --round 0
# The env lives on the node's local disk (see setup_env.sh); AOB_ROOT points at the
# AssetOpsBench checkout.
set -euo pipefail
ROOT=${VTSFM_ROOT:-/local/jc6980_vtsfm}
source "$ROOT/vtsfm-prof/bin/activate"
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export AOB_ROOT=${AOB_ROOT:-$HOME/vtsfm/AssetOpsBench}
cd "$(dirname "$0")/.."
exec python profiling/harness.py "$@"
