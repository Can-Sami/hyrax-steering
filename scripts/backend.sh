#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
PID_FILE="$RUN_DIR/backend.pid"
LOG_FILE="$RUN_DIR/backend.log"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-18000}"
BACKEND_API_KEY="${BACKEND_API_KEY:-}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command> [args]

Commands:
  start                 Start backend in background
  stop                  Stop backend
  restart               Restart backend
  status                Show backend process + health
  logs                  Tail backend logs
  test                  Run basic backend smoke checks
  infer <audio_path>    Run intent inference request with audio file

Env overrides:
  BACKEND_PORT (default: 18000)
  BACKEND_HOST (default: 0.0.0.0)
  BACKEND_API_KEY (optional, maps to x-api-key)
USAGE
}

ensure_venv() {
  if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    echo "Missing .venv. Run: python3 -m venv .venv && . .venv/bin/activate && pip install -e .[dev]"
    exit 1
  fi
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_backend() {
  ensure_venv
  mkdir -p "$RUN_DIR"

  if is_running; then
    echo "Backend already running (PID $(cat "$PID_FILE"))."
    return 0
  fi

  if ss -ltn | grep -q ":$BACKEND_PORT "; then
    echo "Port $BACKEND_PORT is already in use. Set BACKEND_PORT to another port."
    exit 1
  fi

  cd "$ROOT_DIR"
  nohup env \
    STT_ENGINE="${STT_ENGINE:-openai_compatible}" \
    EMBEDDING_ENGINE="${EMBEDDING_ENGINE:-openai_compatible}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8002/v1}" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-local-dev-key}" \
    WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}" \
    EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}" \
    API_KEY="$BACKEND_API_KEY" \
    "$ROOT_DIR/.venv/bin/uvicorn" app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
    >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  sleep 2
  if is_running; then
    echo "Backend started on http://127.0.0.1:$BACKEND_PORT (PID $(cat "$PID_FILE"))"
  else
    echo "Backend failed to start. Check logs: $LOG_FILE"
    exit 1
  fi
}

stop_backend() {
  if ! [[ -f "$PID_FILE" ]]; then
    echo "Backend is not running (no pid file)."
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    sleep 1
  fi
  rm -f "$PID_FILE"
  echo "Backend stopped."
}

status_backend() {
  if is_running; then
    echo "Backend running (PID $(cat "$PID_FILE"))"
  else
    echo "Backend not running"
  fi

  curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/healthz" && echo || true
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/readyz" && echo || true
}

logs_backend() {
  mkdir -p "$RUN_DIR"
  touch "$LOG_FILE"
  tail -f "$LOG_FILE"
}

test_backend() {
  echo "[1/3] /api/healthz"
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/healthz" && echo

  echo "[2/3] /api/readyz"
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/readyz" && echo

  echo "[3/3] /api/v1/intents"
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/v1/intents" && echo

  echo "Backend smoke tests passed."
}

infer_backend() {
  local audio_path="${1:-}"
  if [[ -z "$audio_path" || ! -f "$audio_path" ]]; then
    echo "Provide a valid audio file path: $(basename "$0") infer /absolute/path/sample.wav"
    exit 1
  fi

  local -a headers=()
  if [[ -n "$BACKEND_API_KEY" ]]; then
    headers+=( -H "x-api-key: $BACKEND_API_KEY" )
  fi

  curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/v1/inference/intent" \
    "${headers[@]}" \
    -F "audio_file=@$audio_path" \
    -F "language_hint=tr" \
    -F "channel_id=ivr-01" \
    -F "request_id=req-$(date +%s)" && echo
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start) start_backend ;;
    stop) stop_backend ;;
    restart) stop_backend; start_backend ;;
    status) status_backend ;;
    logs) logs_backend ;;
    test) test_backend ;;
    infer) shift; infer_backend "$@" ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
