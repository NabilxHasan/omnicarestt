#!/usr/bin/env bash
# Launch the STT API server.
#
#   ./run_server.sh              # dev mode: single worker, auto-reload on edit
#   MODE=prod ./run_server.sh    # prod mode: single worker, no reload
#
# Why NOT --workers > 1 (even in prod)
# ------------------------------------
# Each uvicorn worker is a separate process and each loads its own copy of the
# 1.5 GB model into RAM. On the M5 (16 GB) or a T4 GPU (16 GB VRAM), two
# workers means OOM. For horizontal scale, run more replicas of the container
# behind a load balancer instead — that's what Modal / RunPod / Railway do
# natively and it doesn't fight over local RAM.
#
# When we deploy to GPU (Modal / RunPod)
# --------------------------------------
# Set these env vars before invoking (or bake them into the container):
#   export STT_DEVICE=cuda:0          # picked up by engine.py via config
#   export STT_DTYPE=float16          # halves VRAM footprint on GPU
# and use the prod MODE below.

set -euo pipefail

HOST="${HOST:-0.0.0.0}"    # listen on all interfaces; containers need this
PORT="${PORT:-8000}"
MODE="${MODE:-dev}"

cd "$(dirname "$0")"

# Activate the shared bangla-stt venv if we're not already in a venv.
# Deployment images should already have deps globally installed and won't hit this.
if [[ -z "${VIRTUAL_ENV:-}" && -d "/Users/arko/claude/bangla-stt/.venv" ]]; then
    # shellcheck disable=SC1091
    source /Users/arko/claude/bangla-stt/.venv/bin/activate
fi

case "$MODE" in
    dev)
        # --reload watches for file edits and restarts. Nice for iterating,
        # murderously slow if it reloads while the model is loading, so
        # only for local dev.
        exec uvicorn api:app --host "$HOST" --port "$PORT" --reload
        ;;
    prod)
        # --access-log off drops per-request stdout spam (use structured
        # logging middleware if you want request-level observability).
        exec uvicorn api:app --host "$HOST" --port "$PORT" --no-access-log
        ;;
    *)
        echo "MODE must be 'dev' or 'prod', got: $MODE" >&2
        exit 2
        ;;
esac
