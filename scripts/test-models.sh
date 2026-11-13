#!/usr/bin/env bash
# Manual model parity and HTTP smoke checks; each adapter owns its environment.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/test-models.sh [--adapter all|timesfm25|toto2|moirai2]
                            [--profile integration|accuracy|performance]
                            [--gpu INDEX] [--port PORT] [--output NEW_DIRECTORY]
                            [--data-dir DIRECTORY] [--cases ACCURACY_MANIFEST]
                            [--requests 100] [--warmups 5] [--batches 1,4]
                            [--contexts 128,512] [--clients 1,4] [--horizon 96]
                            [--backend-order native,vtsfm]
                            [--forecast-precision fp32|tf32|bf16]
                            [--forecast-execution eager|compile]

Run model integration checks sequentially on an idle Linux NVIDIA GPU.
Defaults: all three adapters, GPU 0, port 18001, and a new UTC/commit directory
under results/model-integration/. No CPU .venv is required. Set up each selected
model environment with scripts/setup.sh --adapter NAME before running.
Integration includes a small pinned AssetOpsBench fixture, downloaded once.
Accuracy scores observed labels; --cases selects an optional prepared fev slice.
Performance compares native HTTP and vLLM with matching request boundaries.
--forecast-execution selects the vTSFM neural core; native references stay eager.
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

adapter=all
profile=integration
gpu=0
port=18001
output=
data_dir=
cases=
requests=100
warmups=5
batches=1,4
contexts=128,512
clients=1,4
horizon=96
backend_order=native,vtsfm
forecast_precision="${VTSFM_TEST_PRECISION:-}"
forecast_execution="${VTSFM_TEST_EXECUTION:-eager}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --adapter|--profile|--gpu|--port|--output|--data-dir|--cases|--requests|--warmups|--batches|--contexts|--clients|--horizon|--backend-order|--forecast-precision|--forecast-execution)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
      case "$1" in
        --adapter) adapter="$2" ;;
        --profile) profile="$2" ;;
        --gpu) gpu="$2" ;;
        --port) port="$2" ;;
        --output) output="$2" ;;
        --data-dir) data_dir="$2" ;;
        --cases) cases="$2" ;;
        --requests) requests="$2" ;;
        --warmups) warmups="$2" ;;
        --batches) batches="$2" ;;
        --contexts) contexts="$2" ;;
        --clients) clients="$2" ;;
        --horizon) horizon="$2" ;;
        --backend-order) backend_order="$2" ;;
        --forecast-precision) forecast_precision="$2" ;;
        --forecast-execution) forecast_execution="$2" ;;
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
case "$profile" in
  integration|accuracy|performance) ;;
  *) fail 'Unsupported profile; use integration, accuracy, or performance' ;;
esac
case "$forecast_execution" in
  eager|compile) ;;
  *) fail '--forecast-execution must be eager or compile' ;;
esac
[[ -z "$cases" || "$profile" == accuracy ]] || fail '--cases requires --profile accuracy'
case "$forecast_precision" in
  ''|fp32|tf32) ;;
  bf16) [[ "$adapter" != all && "$adapter" != timesfm25 ]] || fail 'Native BF16 requires --adapter toto2 or moirai2' ;;
  *) fail '--forecast-precision must be fp32, tf32, or bf16' ;;
esac
[[ "$requests" =~ ^[0-9]+$ && ${#requests} -le 7 && 10#$requests -ge 1 ]] || fail '--requests must be positive'
[[ "$warmups" =~ ^[0-9]+$ && ${#warmups} -le 7 ]] || fail '--warmups must be nonnegative'
[[ "$horizon" =~ ^[0-9]+$ && ${#horizon} -le 4 ]] || fail '--horizon must be in [1,1024]'
(( 10#$horizon >= 1 && 10#$horizon <= 1024 )) || fail '--horizon must be in [1,1024]'
for values in "$batches" "$contexts" "$clients"; do
  [[ "$values" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'Grid values must be comma-separated positive integers'
  IFS=, read -r -a numbers <<< "$values"
  for number in "${numbers[@]}"; do
    [[ ${#number} -le 7 ]] || fail 'Grid value is too large'
    (( 10#$number >= 1 )) || fail 'Grid values must be positive'
  done
done
case "$backend_order" in
  native,vtsfm|vtsfm,native) ;;
  *) fail '--backend-order must be native,vtsfm or vtsfm,native' ;;
esac
if [[ "$profile" == performance ]]; then
  IFS=, read -r -a numbers <<< "$clients"
  for number in "${numbers[@]}"; do
    (( 10#$requests >= 10#$number )) || fail '--requests must be at least every client concurrency'
  done
fi
[[ "$gpu" =~ ^[0-9]+$ ]] || fail '--gpu must be a numeric physical GPU index'
[[ "$port" =~ ^[0-9]+$ && ${#port} -le 5 ]] || fail '--port must be an integer in [1,65535]'
(( 10#$port >= 1 && 10#$port <= 65535 )) || fail '--port must be an integer in [1,65535]'
[[ "$(uname -s)" == Linux ]] || fail 'Model integration checks require Linux and an NVIDIA GPU'

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test_file="tests/test_model_$profile.py"
[[ -f "$root/$test_file" ]] || fail "Missing $test_file"
test_files=("$test_file")
if [[ "$forecast_execution" == compile && "$profile" == integration ]]; then
  test_files+=(tests/test_compiled_forecast_runtime.py)
  [[ -f "$root/${test_files[1]}" ]] || fail "Missing ${test_files[1]}"
fi
if [[ -z "$data_dir" ]]; then data_dir="$root/data";
elif [[ "$data_dir" != /* ]]; then data_dir="$PWD/$data_dir"; fi
if [[ -n "$cases" ]]; then
  if [[ "$cases" != /* ]]; then cases="$PWD/$cases"; fi
  [[ -f "$cases" ]] || fail "Missing accuracy manifest: $cases"
fi
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
  output="$root/results/model-$profile/$(date -u +%Y%m%dT%H%M%SZ)-$revision"
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
export VTSFM_TEST_PROFILE="$profile" VTSFM_TEST_DATA="$data_dir" VTSFM_TEST_CASES="$cases"
export VTSFM_TEST_EXECUTION="$forecast_execution"
if [[ "$forecast_execution" == compile ]]; then export TRITON_DEFAULT_FP_FUSION=0; fi
export VTSFM_PERF_REQUESTS="$requests" VTSFM_PERF_WARMUPS="$warmups"
export VTSFM_PERF_BATCHES="$batches" VTSFM_PERF_CONTEXTS="$contexts"
export VTSFM_PERF_CLIENTS="$clients" VTSFM_PERF_HORIZON="$horizon" VTSFM_PERF_ORDER="$backend_order"
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset VTSFM_PRECISION VTSFM_TEST_PRECISION
if [[ -n "$forecast_precision" ]]; then
  export VTSFM_TEST_PRECISION="$forecast_precision"
  if [[ "$forecast_precision" == tf32 ]]; then export NVIDIA_TF32_OVERRIDE=1; fi
fi
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
  "$root/.venv-$name/bin/python" -m pytest -m gpu "${test_files[@]}" \
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
