#!/usr/bin/env bash
# Manual model parity and HTTP smoke checks; each adapter owns its environment.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/test-models.sh [--adapter all|timesfm25|toto2|moirai2]
                            [--gpu INDEX] [--port PORT] [--output NEW_DIRECTORY]

Run model integration checks sequentially on an idle Linux NVIDIA GPU.
Defaults: all three adapters, GPU 0, port 18001, and a new UTC/commit directory
under results/model-integration/. No CPU .venv is required. Set up each selected
model environment with scripts/setup.sh --adapter NAME before running.
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

adapter=all
gpu=0
port=18001
output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --adapter|--gpu|--port|--output)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
      case "$1" in
        --adapter) adapter="$2" ;;
        --gpu) gpu="$2" ;;
        --port) port="$2" ;;
        --output) output="$2" ;;
      esac
      shift 2 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

case "$adapter" in
  all) adapters=(timesfm25 toto2 moirai2) ;;
  timesfm25|toto2|moirai2) adapters=("$adapter") ;;
  *) fail 'Unsupported adapter; use all, timesfm25, toto2, or moirai2' ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || fail '--gpu must be a numeric physical GPU index'
[[ "$port" =~ ^[0-9]+$ && ${#port} -le 5 ]] || fail '--port must be an integer in [1,65535]'
(( 10#$port >= 1 && 10#$port <= 65535 )) || fail '--port must be an integer in [1,65535]'
[[ "$(uname -s)" == Linux ]] || fail 'Model integration checks require Linux and an NVIDIA GPU'

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ -f "$root/tests/test_model_integration.py" ]] || fail 'Missing tests/test_model_integration.py'
missing=0
for name in "${adapters[@]}"; do
  if [[ ! -x "$root/.venv-$name/bin/python" ]]; then
    printf 'Missing environment for %s. Run: ./scripts/setup.sh --adapter %s\n' "$name" "$name" >&2
    missing=1
  fi
done
[[ "$missing" == 0 ]] || exit 2

if [[ -z "$output" ]]; then
  revision="$(git -C "$root" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
  output="$root/results/model-integration/$(date -u +%Y%m%dT%H%M%SZ)-$revision"
elif [[ "$output" != /* ]]; then
  output="$PWD/$output"
fi
# mkdir is the atomic no-overwrite check, including an existing symlink.
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output" || fail "Output already exists or cannot be created: $output"
output="$(cd -- "$output" && pwd -P)"
summary="$output/summary.tsv"
printf 'adapter\tpytest_exit_code\tlog_exit_code\n' > "$summary"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export VTSFM_TEST_GPU_INDEX="$gpu" VTSFM_TEST_PORT="$port"
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$root/src:$root${PYTHONPATH:+:$PYTHONPATH}"

pytest_pid=
log_pid=
fifo=
active_adapter=
launching=0
pending_cancel=

wait_owned() {
  local pid="$1" attempts="$2" count=0
  while kill -0 "$pid" 2>/dev/null && [[ "$count" -lt "$attempts" ]]; do
    sleep 0.1
    count=$((count + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf 'Cleanup deadline exceeded for owned process %s; terminating it\n' "$pid" >&2
    kill -KILL "$pid" 2>/dev/null || true
    # A broken kernel may leave a process unkillable; do not wait forever.
    count=0
    while kill -0 "$pid" 2>/dev/null && [[ "$count" -lt 10 ]]; do
      sleep 0.1
      count=$((count + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      return 124
    fi
  fi
  wait "$pid"
}

cancel() {
  local exit_code="$1" pytest_status="$1" log_status=0
  # Capture both child PIDs before acting on a signal during launch.
  if [[ "$launching" == 1 ]]; then
    pending_cancel="$exit_code"
    return 0
  fi
  trap '' INT TERM
  set +e
  if [[ -n "$pytest_pid" ]]; then
    # Background jobs may inherit ignored SIGINT. The pytest fixture installs
    # SIGTERM cleanup, so use it for both Ctrl-C and shell-only SIGTERM.
    kill -TERM "$pytest_pid" 2>/dev/null || true
    wait_owned "$pytest_pid" 900
    pytest_status=$?
    pytest_pid=
  fi
  if [[ -n "$log_pid" ]]; then
    wait_owned "$log_pid" 50
    log_status=$?
    log_pid=
  fi
  [[ -z "$fifo" ]] || rm -f -- "$fifo"
  if [[ -n "$active_adapter" ]]; then
    printf '%s\t%s\t%s\n' "$active_adapter" "$pytest_status" "$log_status" >> "$summary"
  fi
  exit "$exit_code"
}

trap 'cancel 130' INT
trap 'cancel 143' TERM
cd -- "$root"

failed=0
for name in "${adapters[@]}"; do
  folder="$output/$name"
  mkdir -- "$folder"
  export VTSFM_TEST_ADAPTER="$name" VTSFM_TEST_OUTPUT="$folder"
  printf 'Running %s; artifacts: %s\n' "$name" "$folder"
  active_adapter="$name"
  launching=1
  fifo="$folder/.pytest-output.fifo"
  mkfifo -- "$fifo"
  tee "$folder/pytest.log" < "$fifo" &
  log_pid=$!
  "$root/.venv-$name/bin/python" -m pytest -m gpu tests/test_model_integration.py \
    -q -s --junitxml "$folder/junit.xml" > "$fifo" 2>&1 &
  pytest_pid=$!
  launching=0
  if [[ -n "$pending_cancel" ]]; then
    cancel "$pending_cancel"
  fi
  set +e
  wait "$pytest_pid"
  pytest_status=$?
  pytest_pid=
  wait "$log_pid"
  log_status=$?
  log_pid=
  set -e
  rm -- "$fifo"
  fifo=
  printf '%s\t%s\t%s\n' "$name" "$pytest_status" "$log_status" >> "$summary"
  active_adapter=
  if [[ "$pytest_status" == 130 || "$pytest_status" == 143 ]]; then
    exit "$pytest_status"
  fi
  if [[ "$pytest_status" != 0 || "$log_status" != 0 ]]; then
    failed=1
  fi
done
printf 'Summary: %s\n' "$summary"
exit "$failed"
