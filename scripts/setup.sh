#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup.sh [--cpu [--evaluation]] [--adapter chronos2|timesfm25|toto2|moirai2]

Create this clone's .venv with Python 3.11.17 and install vTSFM editable.
  --cpu       Install tests, data preparation, and reporting without Torch/vLLM.
  --evaluation Add pinned fev preparation tools to the CPU environment only.
  --adapter   Install a new model in .venv-<adapter>; Chronos keeps .venv.
  -h, --help  Show this help without downloading or changing anything.

GPU setup requires Linux. The default installs the legacy Chronos stack.
TimesFM, Toto, and Moirai use the vLLM 0.33.0 / CUDA 13.0 wheel stack.
If uv is unavailable or older than 0.13.4, install uv 0.13.4 locally using
Astral's official installer. Older uv versions lack the pinned Python metadata.
Existing UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, UV_PYTHON_BIN_DIR, and uv installer
directory settings are honored; defaults stay inside this checkout.
No drivers, services, shell profiles, or global Python packages are modified.
EOF
}

cpu=0
evaluation=0
adapter=chronos2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) cpu=1 ;;
    --evaluation) evaluation=1 ;;
    --adapter) adapter="${2:?--adapter requires a value}"; shift ;;
    --adapter=*) adapter="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown setup option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
case "$adapter" in
  chronos2|timesfm25|toto2|moirai2) ;;
  *) printf 'Unknown adapter: %s\n' "$adapter" >&2; exit 2 ;;
esac
if [[ "$cpu" == 1 && "$adapter" != chronos2 ]]; then
  printf 'Use --cpu without --adapter for the shared CPU test environment.\n' >&2
  exit 2
fi
if [[ "$evaluation" == 1 && "$cpu" != 1 ]]; then
  printf 'Use --evaluation with --cpu; model environments keep their own pins.\n' >&2
  exit 2
fi
vtsfm_select_environment --adapter "$adapter"
environment_dir="${VTSFM_PYTHON%/bin/python}"
cd -- "$VTSFM_ROOT"
if [[ "$cpu" == 0 && "$(uname -s)" != Linux ]]; then
  printf 'GPU setup requires Linux. Use scripts/setup.sh --cpu on this platform.\n' >&2
  exit 1
fi

python_version=3.11.17
if [[ "$adapter" == toto2 ]]; then python_version=3.12.12; fi
if [[ -x "$VTSFM_PYTHON" ]]; then
  installed_version="$("$VTSFM_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  if [[ "$installed_version" != "$python_version" ]]; then
    printf 'Existing .venv uses Python %s; vTSFM requires %s.\n' "$installed_version" "$python_version" >&2
    printf 'Move or remove that .venv explicitly, then rerun setup; it has not been replaced.\n' >&2
    exit 1
  fi
  if [[ "$cpu" == 0 && "$adapter" != chronos2 ]] &&
    "$VTSFM_PYTHON" -c 'from importlib.metadata import version; version("nvidia-cuda-runtime-cu12")' 2>/dev/null; then
    printf 'Existing environment has CUDA 12 packages: %s\n' "$environment_dir" >&2
    printf '%s requires CUDA 13.0 wheels. Move the environment aside or use a fresh checkout, then rerun setup.\n' "$adapter" >&2
    exit 1
  fi
elif [[ -e "$environment_dir" ]]; then
  printf 'Existing .venv has no working Python. Move or remove it explicitly, then rerun setup.\n' >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$VTSFM_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$VTSFM_ROOT/.tools/python}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-$VTSFM_ROOT/.tools/python-bin}"
uv_install_dir="${UV_UNMANAGED_INSTALL:-${UV_INSTALL_DIR:-$VTSFM_ROOT/.tools/uv}}"
uv_is_current() {
  local version
  version="$("$1" --version 2>/dev/null)" || return 1
  if [[ "$version" =~ ^uv\ ([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    (( BASH_REMATCH[1] > 0 || BASH_REMATCH[2] > 13 ||
       (BASH_REMATCH[2] == 13 && BASH_REMATCH[3] >= 4) ))
  else
    return 1
  fi
}
if command -v uv >/dev/null 2>&1 && uv_is_current "$(command -v uv)"; then
  uv_bin="$(command -v uv)"
elif [[ -x "$uv_install_dir/uv" ]] && uv_is_current "$uv_install_dir/uv"; then
  uv_bin="$uv_install_dir/uv"
else
  if ! command -v curl >/dev/null 2>&1; then
    printf 'curl is required to download uv. Install curl or uv, then rerun setup.\n' >&2
    exit 1
  fi
  installer="$(mktemp "${TMPDIR:-/tmp}/vtsfm-uv-installer.XXXXXX")"
  trap 'rm -f -- "$installer"' EXIT
  curl --fail --silent --show-error --location \
    https://astral.sh/uv/0.13.4/install.sh --output "$installer"
  UV_UNMANAGED_INSTALL="$uv_install_dir" UV_INSTALL_DIR="$uv_install_dir" \
    UV_NO_MODIFY_PATH=1 sh "$installer"
  uv_bin="$uv_install_dir/uv"
  if [[ ! -x "$uv_bin" ]]; then
    printf 'uv installation did not produce %s. Inspect the installer output and retry.\n' "$uv_bin" >&2
    exit 1
  fi
fi

if [[ ! -x "$VTSFM_PYTHON" ]]; then
  "$uv_bin" python install "$python_version"
  "$uv_bin" venv --python "$python_version" "$environment_dir"
fi
install_args=(pip install --python "$VTSFM_PYTHON")
extras='.[serve-chronos,dev,benchmark]'
if [[ "$adapter" != chronos2 ]]; then
  extras=".[serve,dev,$adapter]"
  install_args+=(--torch-backend cu130)
  if [[ "$adapter" == moirai2 ]]; then
    install_args+=(--override "$VTSFM_ROOT/scripts/moirai2-overrides.txt")
  fi
fi
if [[ "$cpu" == 1 ]]; then
  extras='.[dev,benchmark]'
  if [[ "$evaluation" == 1 ]]; then extras='.[dev,benchmark,evaluation]'; fi
fi
"$uv_bin" "${install_args[@]}" --editable "$extras"
printf 'Environment ready: %s\n' "$VTSFM_PYTHON"
if [[ "$cpu" == 1 ]]; then
  printf 'Next: scripts/doctor.sh --cpu\n'
else
  printf 'Next: scripts/doctor.sh --adapter %s\n' "$adapter"
fi
