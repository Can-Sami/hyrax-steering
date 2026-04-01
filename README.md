# CallSteering Backend

Backend service for IVR intent recognition.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

## Local Docker dev stack (single script)

Use one script to control the full stack (postgres + model services + backend):

```bash
scripts/dev-stack.sh start
scripts/dev-stack.sh status
scripts/dev-stack.sh logs all
scripts/dev-stack.sh restart backend
scripts/dev-stack.sh stop all
```

If port `18000` (backend) or `8002` (model gateway) is already in use, run with custom host ports:

```bash
BACKEND_HOST_PORT=18010 MODEL_GATEWAY_HOST_PORT=8012 scripts/dev-stack.sh start
```

Default mode is CPU embeddings (`EMBEDDING_BACKEND_MODE=transformers_cpu`).
For GPU/vLLM mode:

```bash
EMBEDDING_BACKEND_MODE=vllm scripts/dev-stack.sh start
```

Health endpoints:
- `GET /api/healthz`
- `GET /api/readyz`

## Database migrations (Step 2)

Set a database URL (optional if default local postgres matches):

```bash
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/callsteering"
```

Run migrations:

```bash
alembic upgrade head
```

## Intent APIs (POC)

- `GET /api/v1/intents`
- `POST /api/v1/intents` with JSON `{ "intent_code": "...", "description": "..." }`
- `PUT /api/v1/intents/{intent_id}` with JSON `{ "intent_code": "...", "description": "..." }`
- `DELETE /api/v1/intents/{intent_id}`
- `POST /api/v1/intents/search` with JSON `{ "query": "...", "k": 5 }`

`POST /api/v1/intents/search` performs semantic similarity search using pgvector.
