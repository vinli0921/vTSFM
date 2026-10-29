#!/usr/bin/env bash
# Shared path handling; sourcing this file does not create an environment.
VTSFM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VTSFM_PYTHON="$VTSFM_ROOT/.venv/bin/python"

vtsfm_select_environment() {
  local adapter=chronos2
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --adapter) adapter="${2:-}"; shift ;;
      --adapter=*) adapter="${1#*=}" ;;
    esac
    shift
  done
  case "$adapter" in
    timesfm25|toto2|moirai2) VTSFM_PYTHON="$VTSFM_ROOT/.venv-$adapter/bin/python" ;;
  esac
}

vtsfm_require_environment() {
  if [[ ! -x "$VTSFM_PYTHON" ]]; then
    printf 'Missing project Python environment: %s\n' "$VTSFM_PYTHON" >&2
    printf 'Run scripts/setup.sh --cpu for CPU tools, or scripts/setup.sh on a Linux GPU host.\n' >&2
    return 1
  fi
  cd -- "$VTSFM_ROOT"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
}
