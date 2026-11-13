#!/usr/bin/env bash
# Issue #16 Nsight runs: in-process (native, runtime) and HTTP, traced plus untraced timing.
# Usage: benchmarks/nsight/run.sh OUT_DIR [ADAPTER...]   (default adapters: timesfm25 toto2 moirai2)
# Env: NSYS (nsys binary), AOB_ROOT (AssetOpsBench checkout with chiller_6.json),
#      METROPT_DATA (prepared MetroPT dir, enables the TimesFM block-mode case), HF_HOME.
# Each run waits while another process holds more than 1 GiB on the GPU. HTTP runs refuse an occupied
# port and signal only the process group they started. Any failed stage makes the script exit nonzero.
set -u
shopt -s nullglob
command -v setsid >/dev/null || { echo "setsid (util-linux) is required" >&2; exit 2; }
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$here/../.." && pwd -P)"
out="${1:?usage: run.sh OUT_DIR [ADAPTER...]}"; shift
adapters=("$@"); [[ $# -eq 0 ]] && adapters=(timesfm25 toto2 moirai2)
NSYS="${NSYS:-nsys}"; export NSYS
port=18516
mkdir -p "$out"/{traces,results,logs,stats}
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
declare -A model_id=([timesfm25]=google/timesfm-2.5-200m-pytorch [toto2]=Datadog/Toto-2.0-313m
                     [moirai2]=Salesforce/moirai-2.0-R-small)
declare -A patch=([timesfm25]=32 [toto2]=32 [moirai2]=16)

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/logs/progress.log"; }
failed=0
fail() { log "FAILED $*"; failed=1; }
port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; }
gpu_wait() {
  while nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits |
      awk '$1+0 > 1024 {f=1} END {exit !f}'; do
    log "waiting: another process holds GPU memory"; sleep 60; done
}

inproc() { # ADAPTER PATH ROUND N
  local a=$1 p=$2 r=$3 n=$4 tag=${1}_${2}_r$3 py="$root/.venv-$1/bin/python"
  gpu_wait; log "start $tag"
  PYTHONPATH="$root/src" "$NSYS" profile --trace=cuda,nvtx,osrt,python-gil --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
    -o "$out/traces/$tag" "$py" "$here/driver.py" --model "$a" --path "$p" --round "$r" --n "$n" \
    --out "$out/results/$tag.json" > "$out/logs/$tag.log" 2>&1 && [[ -s "$out/results/$tag.json" ]] || fail "$tag"
  gpu_wait
  PYTHONPATH="$root/src" "$py" "$here/driver.py" --model "$a" --path "$p" --round "$r" --n "$n" \
    --out "$out/results/${tag}_untraced.json" > "$out/logs/${tag}_untraced.log" 2>&1 || fail "${tag}_untraced"
}

http_once() { # ADAPTER MODE ROUND N TAG TRACED
  local a=$1 m=$2 r=$3 n=$4 tag=$5 traced=$6 py="$root/.venv-$1/bin/python" session="vtsfm_$5"
  local launch=() client=()
  [[ $traced == 1 ]] && client=(--session "$session" --trace-out "$out/traces/$tag") || launch=(--no-nsys)
  gpu_wait; log "start $tag"
  if port_in_use; then fail "$tag: port $port is already in use; not starting a server"; return; fi
  # setsid gives the server (launch_server.py execs nsys/vLLM, keeping this PID) its own process
  # group, so cleanup signals only processes this run started.
  setsid "$py" "$here/launch_server.py" --adapter "$a" --mode "$m" --port $port --session "$session" \
    "${launch[@]}" > "$out/logs/server_$tag.log" 2>&1 &
  local server=$! ready=0
  for _ in $(seq 1 200); do
    kill -0 "$server" 2>/dev/null || break
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ready=1; break; }
    sleep 3
  done
  if [[ $ready == 1 ]]; then
    PYTHONPATH="$root/src:$root" "$py" "$here/http_client.py" --url "http://127.0.0.1:$port" \
      --model-id "${model_id[$a]}" --round "$r" --patch "${patch[$a]}" --n "$n" \
      ${METROPT_DATA:+--data-dir "$METROPT_DATA"} "${client[@]}" --out "$out/results/$tag.json" \
      > "$out/logs/client_$tag.log" 2>&1 || fail "$tag"
  else
    fail "$tag: server did not become healthy"
  fi
  [[ $traced == 1 ]] && { "$NSYS" shutdown --session="$session" >/dev/null 2>&1 & }
  sleep 3; kill -INT -- "-$server" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 -- "-$server" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$server" 2>/dev/null
  wait "$server" 2>/dev/null
}
http() { http_once "$1" "$2" "$3" "$4" "${1}_http-${2}_r${3}" 1; http_once "$1" "$2" "$3" "$4" "${1}_http-${2}_r${3}_untraced" 0; }

for a in "${adapters[@]}"; do
  inproc "$a" native 1 50; inproc "$a" runtime 1 50
  inproc "$a" native 0b 30; inproc "$a" runtime 0b 30
  inproc "$a" native 0b16 30; inproc "$a" runtime 0b16 30
  http "$a" single 1 50
  # TimesFM admission caps HTTP requests at 16 rows, so it has no 64-row HTTP round.
  [[ $a == timesfm25 ]] || http "$a" single 0b 30
  http "$a" single 0b16 30
  if [[ $a == timesfm25 && -n "${METROPT_DATA:-}" ]]; then
    http "$a" continuous sat 50; http "$a" whole sat 50
  fi
done

for rep in "$out"/traces/*.nsys-rep; do
  t=$(basename "$rep" .nsys-rep)
  "$NSYS" export --type sqlite -o "$out/traces/$t.sqlite" --force-overwrite true "$rep" \
    > "$out/logs/export_$t.log" 2>&1 || { fail "export $t"; continue; }
  "$NSYS" stats -q --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum --format csv \
    --force-overwrite true -o "$out/stats/$t" "$out/traces/$t.sqlite" > "$out/logs/stats_$t.log" 2>&1 \
    || fail "stats $t"
done
sqlite=("$out"/traces/*.sqlite)
if [[ ${#sqlite[@]} -eq 0 ]]; then
  fail "no traces to analyze"
elif ! python3 "$here/analyze.py" "${sqlite[@]}" > "$out/results/analysis.jsonl"; then
  fail "analysis"
elif grep -q '"error"' "$out/results/analysis.jsonl"; then
  fail "analysis: some traces have no request windows (see analysis.jsonl)"
fi
if [[ $failed != 0 ]]; then
  log "finished with failures; see FAILED lines above and $out/logs"
  exit 1
fi
log "done: $out/results/analysis.jsonl"
