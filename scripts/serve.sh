#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
vtsfm_select_environment "$@"
vtsfm_require_environment
exec "$VTSFM_PYTHON" -m vtsfm.cli serve "$@"
