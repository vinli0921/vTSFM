#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: scripts/request.sh [URL]\nDefault: http://127.0.0.1:8001/pooling\n'
  printf 'Accepts a base URL or a URL ending in /pooling. Sends examples/request.json.\n'
  exit 0
fi
if [[ $# -gt 1 ]]; then
  printf 'Usage: scripts/request.sh [URL]\n' >&2
  exit 2
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
url="${1:-http://127.0.0.1:8001}"
if [[ "$url" != http://* && "$url" != https://* ]]; then
  printf 'Request URL must start with http:// or https://.\n' >&2
  exit 2
fi
url="${url%/}"
if [[ "$url" != */pooling ]]; then
  url="$url/pooling"
fi
if ! command -v curl >/dev/null 2>&1; then
  printf 'curl is required to send the example request.\n' >&2
  exit 1
fi
if [[ ! -f "$VTSFM_ROOT/examples/request.json" ]]; then
  printf 'Example payload missing: %s/examples/request.json\n' "$VTSFM_ROOT" >&2
  exit 1
fi
exec curl --fail-with-body --silent --show-error --connect-timeout 10 --max-time 180 \
  --header 'Content-Type: application/json' \
  --data-binary "@$VTSFM_ROOT/examples/request.json" --write-out '\n' --url "$url"
