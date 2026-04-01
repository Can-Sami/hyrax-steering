#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
COMPOSE="docker compose -f $COMPOSE_FILE"
MODE=""
MODEL_PROFILE=""
BACKEND_HOST_PORT="${BACKEND_HOST_PORT:-18000}"
MODEL_GATEWAY_HOST_PORT="${MODEL_GATEWAY_HOST_PORT:-8002}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command> [component]

Commands:
  start [all|backend|postgres|model-gateway|embeddings-cpu|embedding-vllm]
  stop [all|backend|postgres|model-gateway|embeddings-cpu|embedding-vllm]
  restart [all|backend|postgres|model-gateway|embeddings-cpu|embedding-vllm]
  status
  logs [all|backend|postgres|model-gateway|embeddings-cpu|embedding-vllm]
  test

Environment:
  EMBEDDING_BACKEND_MODE=vllm|transformers_cpu (default: transformers_cpu)
  BACKEND_HOST_PORT (default: 18000)
  MODEL_GATEWAY_HOST_PORT (default: 8002)
USAGE
}

require_compose() {
  command -v docker >/dev/null 2>&1 || { echo "docker not found"; exit 1; }
  docker compose version >/dev/null 2>&1 || { echo "docker compose not available"; exit 1; }
}

resolve_mode() {
  MODE="${EMBEDDING_BACKEND_MODE:-transformers_cpu}"
  case "$MODE" in
    vllm)
      MODEL_PROFILE="vllm"
      ;;
    transformers_cpu)
      MODEL_PROFILE="cpu"
      ;;
    *)
      echo "Invalid EMBEDDING_BACKEND_MODE: '$MODE'. Expected 'vllm' or 'transformers_cpu'."
      exit 1
      ;;
  esac
}

port_in_use() {
  local port="$1"
  ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"
}

print_port_help() {
  local port="$1"
  local env_name="$2"
  cat <<EOF
Port $port is already in use on host.
Either stop the process using it, or run with a different port:

  $env_name=<new_port> scripts/dev-stack.sh start

Example:
  BACKEND_HOST_PORT=18010 scripts/dev-stack.sh start
EOF
}

preflight_ports() {
  local target="${1:-all}"

  if [[ "$target" == "all" || "$target" == "backend" ]]; then
    if port_in_use "$BACKEND_HOST_PORT"; then
      print_port_help "$BACKEND_HOST_PORT" "BACKEND_HOST_PORT"
      exit 1
    fi
  fi

  if [[ "$target" == "all" || "$target" == "model-gateway" ]]; then
    if port_in_use "$MODEL_GATEWAY_HOST_PORT"; then
      print_port_help "$MODEL_GATEWAY_HOST_PORT" "MODEL_GATEWAY_HOST_PORT"
      exit 1
    fi
  fi
}

compose_up() {
  local target="${1:-all}"
  preflight_ports "$target"
  if [[ "$target" == "all" ]]; then
    $COMPOSE --profile "$MODEL_PROFILE" up -d --build
    return
  fi
  $COMPOSE --profile "$MODEL_PROFILE" up -d --build "$target"
}

compose_stop() {
  local target="${1:-all}"
  if [[ "$target" == "all" ]]; then
    $COMPOSE down
    return
  fi
  $COMPOSE stop "$target"
}

compose_restart() {
  local target="${1:-all}"
  if [[ "$target" == "all" ]]; then
    $COMPOSE --profile "$MODEL_PROFILE" down
    $COMPOSE --profile "$MODEL_PROFILE" up -d --build
    return
  fi
  $COMPOSE --profile "$MODEL_PROFILE" restart "$target"
}

compose_logs() {
  local target="${1:-all}"
  if [[ "$target" == "all" ]]; then
    $COMPOSE --profile "$MODEL_PROFILE" logs --no-color -f --tail=200
    return
  fi
  $COMPOSE --profile "$MODEL_PROFILE" logs --no-color -f --tail=200 "$target"
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
  wait_http_ok "backend" "http://127.0.0.1:${BACKEND_HOST_PORT}/api/readyz" 60 2
  wait_http_ok "model-gateway" "http://127.0.0.1:${MODEL_GATEWAY_HOST_PORT}/readyz" 60 2

  echo "[1/4] backend /api/healthz"
  curl -fsS "http://127.0.0.1:${BACKEND_HOST_PORT}/api/healthz" && echo

  echo "[2/4] backend /api/v1/intents"
  curl -fsS "http://127.0.0.1:${BACKEND_HOST_PORT}/api/v1/intents" && echo

  echo "[3/4] model-gateway /readyz"
  curl -fsS "http://127.0.0.1:${MODEL_GATEWAY_HOST_PORT}/readyz" && echo

  echo "[4/4] model-gateway /v1/embeddings"
  curl -fsS "http://127.0.0.1:${MODEL_GATEWAY_HOST_PORT}/v1/embeddings" \
    -H "Authorization: Bearer ${GATEWAY_API_KEY:-local-dev-key}" \
    -H "Content-Type: application/json" \
    -d '{"input":"merhaba"}' | head -c 300 && echo
}

main() {
  local cmd="${1:-}"
  local target="${2:-all}"

  if [[ -z "$cmd" ]]; then
    usage
    exit 1
  fi

  require_compose
  resolve_mode

  case "$cmd" in
    start) compose_up "$target" ;;
    stop) compose_stop "$target" ;;
    restart) compose_restart "$target" ;;
    status) $COMPOSE --profile "$MODEL_PROFILE" ps ;;
    logs) compose_logs "$target" ;;
    test) smoke_test ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
