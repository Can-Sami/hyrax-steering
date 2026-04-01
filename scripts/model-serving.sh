#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MS_DIR="$ROOT_DIR/model-serving"
COMPOSE="docker compose"
MODE=""
COMPOSE_PROFILE=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command>

Commands:
  start      Build and start model-serving stack in background
  stop       Stop and remove stack
  restart    Restart stack
  update     Pull latest images and restart stack
  status     Show service status
  logs       Tail logs (all services)
  test       Run smoke checks (with readiness wait)
USAGE
}

require_compose() {
  command -v docker >/dev/null 2>&1 || { echo "docker not found"; exit 1; }
  docker compose version >/dev/null 2>&1 || { echo "docker compose not available"; exit 1; }
}

resolve_mode() {
  local env_file="$MS_DIR/.env"
  local mode="${EMBEDDING_BACKEND_MODE:-}"

  if [[ -z "$mode" && -f "$env_file" ]]; then
    mode="$(grep -E '^EMBEDDING_BACKEND_MODE=' "$env_file" | head -n1 | cut -d= -f2- | tr -d '[:space:]')"
  fi

  mode="${mode#\"}"
  mode="${mode%\"}"
  mode="${mode#\'}"
  mode="${mode%\'}"

  if [[ -z "$mode" ]]; then
    mode="vllm"
  fi

  case "$mode" in
    vllm)
      COMPOSE_PROFILE="vllm"
      ;;
    transformers_cpu)
      COMPOSE_PROFILE="cpu"
      ;;
    *)
      echo "Invalid EMBEDDING_BACKEND_MODE: '$mode'. Expected 'vllm' or 'transformers_cpu'."
      exit 1
      ;;
  esac

  MODE="$mode"
}

validate_env_constraints() {
  [[ "$MODE" == "vllm" ]] || return 0

  local env_file="$MS_DIR/.env"
  local model=""
  local max_len=""

  if [[ -f "$env_file" ]]; then
    model="$(grep -E '^EMBEDDING_MODEL_NAME=' "$env_file" | head -n1 | cut -d= -f2-)"
    max_len="$(grep -E '^VLLM_MAX_MODEL_LEN=' "$env_file" | head -n1 | cut -d= -f2-)"
  fi

  if [[ -z "$model" ]]; then
    model="BAAI/bge-small-en-v1.5"
  fi

  if [[ -z "$max_len" ]]; then
    return 0
  fi

  if [[ "$model" == "BAAI/bge-small-en-v1.5" ]] && [[ "$max_len" -gt 512 ]]; then
    echo "Invalid .env config: BAAI/bge-small-en-v1.5 supports max 512, but VLLM_MAX_MODEL_LEN=$max_len"
    echo "Fix: set VLLM_MAX_MODEL_LEN=512 (or switch embedding model)."
    exit 1
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_mem_mb=""
    gpu_mem_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
    if [[ -n "$gpu_mem_mb" ]] && [[ "$model" == "jinaai/jina-embeddings-v3" ]] && [[ "$gpu_mem_mb" -lt 8000 ]]; then
      echo "Model $model is not supported on this GPU (${gpu_mem_mb}MB)."
      echo "Observed crash is CUDA OOM during model load. Use BAAI/bge-small-en-v1.5 on this machine or run jina model on a larger GPU."
      exit 1
    fi
  fi
}

start_stack() {
  cd "$MS_DIR"
  $COMPOSE --profile "$COMPOSE_PROFILE" up -d --build
  $COMPOSE --profile "$COMPOSE_PROFILE" ps
}

stop_stack() {
  cd "$MS_DIR"
  $COMPOSE down
}

restart_stack() {
  stop_stack
  start_stack
}

update_stack() {
  cd "$MS_DIR"
  $COMPOSE --profile "$COMPOSE_PROFILE" pull || true
  $COMPOSE --profile "$COMPOSE_PROFILE" up -d --build
  $COMPOSE --profile "$COMPOSE_PROFILE" ps
}

status_stack() {
  cd "$MS_DIR"
  $COMPOSE --profile "$COMPOSE_PROFILE" ps
}

logs_stack() {
  cd "$MS_DIR"
  $COMPOSE --profile "$COMPOSE_PROFILE" logs --no-color -f --tail=200
}

wait_http_ok() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"
  local delay="${4:-2}"

  echo "Waiting for $name at $url ..."
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS -m 5 "$url" >/dev/null; then
      echo "$name is ready"
      return 0
    fi
    sleep "$delay"
  done

  echo "Timed out waiting for $name"
  return 1
}

smoke_test() {
  wait_http_ok "gateway" "http://127.0.0.1:8002/readyz" 30 2
  if [[ "$MODE" == "vllm" ]]; then
    wait_http_ok "vllm" "http://127.0.0.1:18001/v1/models" 120 3
  else
    wait_http_ok "embeddings-cpu" "http://127.0.0.1:18003/v1/models" 120 3
  fi

  echo "[1/4] gateway /healthz"
  curl -fsS http://127.0.0.1:8002/healthz && echo

  echo "[2/4] gateway /readyz"
  curl -fsS http://127.0.0.1:8002/readyz && echo

  if [[ "$MODE" == "vllm" ]]; then
    echo "[3/4] vLLM /v1/models"
    curl -fsS http://127.0.0.1:18001/v1/models && echo
  else
    echo "[3/4] embeddings-cpu /v1/models"
    curl -fsS http://127.0.0.1:18003/v1/models && echo
  fi

  echo "[4/4] gateway embeddings"
  local gateway_embeddings
  gateway_embeddings="$(curl -fsS http://127.0.0.1:8002/v1/embeddings \
    -H "Authorization: Bearer local-dev-key" \
    -H "Content-Type: application/json" \
    -d '{"input":"merhaba"}')"
  printf '%s\n' "$gateway_embeddings" | head -c 400 && echo

  echo "Smoke tests passed."
}

main() {
  local cmd="${1:-}"
  if [[ -z "$cmd" ]]; then
    usage
    exit 1
  fi

  require_compose
  resolve_mode
  validate_env_constraints

  case "$cmd" in
    start) start_stack ;;
    stop) stop_stack ;;
    restart) restart_stack ;;
    update) update_stack ;;
    status) status_stack ;;
    logs) logs_stack ;;
    test) smoke_test ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
