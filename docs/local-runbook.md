# Local Runbook

1. For full local Docker stack, run `scripts/dev-stack.sh start`.
2. Check status with `scripts/dev-stack.sh status`.
3. Tail logs with `scripts/dev-stack.sh logs all` (or `logs backend`, `logs postgres`, etc.).
4. Restart components with `scripts/dev-stack.sh restart backend` or `scripts/dev-stack.sh restart all`.
5. Stop stack with `scripts/dev-stack.sh stop all`.
6. If host ports are occupied, override with:
   - `BACKEND_HOST_PORT=18010`
   - `MODEL_GATEWAY_HOST_PORT=8012`
