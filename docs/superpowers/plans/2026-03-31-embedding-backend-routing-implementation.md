# Embedding Backend Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit embedding backend routing (`vllm` or `transformers_cpu`) behind the existing model gateway without changing backend API usage.

**Architecture:** Keep `model-gateway` as the only public OpenAI-compatible facade. Introduce a small routing layer that selects exactly one upstream by `EMBEDDING_BACKEND_MODE`, then normalize output into one response contract. Add a separate CPU embedding service in compose and mode-aware operational scripts while preserving vLLM-first production defaults.

**Tech Stack:** FastAPI, httpx, sentence-transformers, Docker Compose, Bash, pytest.

---

### Task 1: Lock behavior with failing tests first

**Files:**
- Modify: `tests/model_serving/test_gateway.py`
- Test: `tests/model_serving/test_gateway.py`

- [ ] **Step 1: Add failing test for strict mode routing (`vllm` mode must not fallback)**

```python
def test_embeddings_vllm_mode_upstream_failure_returns_503(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'vllm')
    module = load_gateway_app(monkeypatch)

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError('connect fail')

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: FailingClient())
    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )
    assert response.status_code == 503
```

- [ ] **Step 2: Add failing test for mode validation**

```python
def test_invalid_embedding_backend_mode_returns_400(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'invalid_mode')
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )
    assert response.status_code == 400
    assert 'EMBEDDING_BACKEND_MODE' in response.json()['detail']
```

- [ ] **Step 3: Add failing test for transformers CPU mode forwarding target**

```python
def test_embeddings_transformers_cpu_mode_targets_cpu_service(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'transformers_cpu')
    module = load_gateway_app(monkeypatch)
    captured = {}

    class CapturingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            captured['url'] = url
            return httpx.Response(
                status_code=200,
                request=httpx.Request('POST', url),
                json={'object': 'list', 'data': [{'embedding': [0.1], 'index': 0}], 'model': 'cpu-model', 'usage': {'prompt_tokens': 1, 'total_tokens': 1}},
            )

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: CapturingClient())
    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )
    assert response.status_code == 200
    assert '/v1/embeddings' in captured['url']
    assert 'embeddings-cpu' in captured['url']
```

- [ ] **Step 4: Run focused tests to confirm failures**

Run:
```bash
. .venv/bin/activate && pytest tests/model_serving/test_gateway.py -q
```

Expected: new tests fail (status/target mismatches) before implementation.

- [ ] **Step 5: Commit failing-test checkpoint**

```bash
git add tests/model_serving/test_gateway.py
git commit -m "test: lock embedding mode routing behavior"
```

### Task 2: Implement gateway routing abstraction with explicit modes

**Files:**
- Create: `model-serving/gateway/embedding_router.py`
- Modify: `model-serving/gateway/app.py`
- Test: `tests/model_serving/test_gateway.py`

- [ ] **Step 1: Create router module with strict mode resolution**

```python
# model-serving/gateway/embedding_router.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    mode: str
    base_url: str
    api_key: str
    default_model: str


def resolve_embedding_backend(mode: str, vllm_url: str, cpu_url: str, api_key: str, model: str) -> EmbeddingBackendConfig:
    if mode == 'vllm':
        return EmbeddingBackendConfig(mode=mode, base_url=vllm_url.rstrip('/'), api_key=api_key, default_model=model)
    if mode == 'transformers_cpu':
        return EmbeddingBackendConfig(mode=mode, base_url=cpu_url.rstrip('/'), api_key=api_key, default_model=model)
    raise ValueError('Invalid EMBEDDING_BACKEND_MODE. Use vllm or transformers_cpu.')
```

- [ ] **Step 2: Update gateway embedding flow to use router and return 503 on upstream unavailability**

```python
# model-serving/gateway/app.py (embedding-related excerpt)
from embedding_router import resolve_embedding_backend

EMBEDDING_BACKEND_MODE = os.getenv('EMBEDDING_BACKEND_MODE', 'vllm')
EMBEDDING_VLLM_BASE_URL = os.getenv('EMBEDDING_VLLM_BASE_URL', 'http://embedding-vllm:8000/v1')
EMBEDDING_CPU_BASE_URL = os.getenv('EMBEDDING_CPU_BASE_URL', 'http://embeddings-cpu:8000/v1')

...

@app.post('/v1/embeddings')
async def embeddings(payload: dict, authorization: str | None = Header(default=None)) -> JSONResponse:
    check_auth(authorization)
    if 'input' not in payload:
        raise HTTPException(status_code=400, detail='Request must include input.')
    try:
        backend = resolve_embedding_backend(
            mode=EMBEDDING_BACKEND_MODE,
            vllm_url=EMBEDDING_VLLM_BASE_URL,
            cpu_url=EMBEDDING_CPU_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            model=DEFAULT_EMBEDDING_MODEL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request_body = {'model': DEFAULT_EMBEDDING_MODEL, 'input': payload.get('input')}
    headers = {'Authorization': f'Bearer {backend.api_key}'}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f'{backend.base_url}/embeddings', json=request_body, headers=headers)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail=f'Embedding backend unavailable: {backend.mode}.') from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail=f'Embedding backend timeout: {backend.mode}.') from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail='Embedding upstream failed.') from exc

    body = response.json()
    body['backend_used'] = backend.mode
    return JSONResponse(body)
```

- [ ] **Step 3: Add mode-aware readiness endpoint behavior**

```python
@app.get('/readyz')
def readyz(backend: str | None = None) -> dict[str, str]:
    if backend and backend not in {'vllm', 'transformers_cpu'}:
        raise HTTPException(status_code=400, detail='backend must be vllm or transformers_cpu')
    return {'status': 'ready', 'service': 'callsteering-model-gateway', 'embedding_mode': EMBEDDING_BACKEND_MODE}
```

- [ ] **Step 4: Run tests and ensure new behaviors pass**

Run:
```bash
. .venv/bin/activate && pytest tests/model_serving/test_gateway.py -q
```

Expected: all gateway tests pass, including new mode tests.

- [ ] **Step 5: Commit gateway routing implementation**

```bash
git add model-serving/gateway/app.py model-serving/gateway/embedding_router.py tests/model_serving/test_gateway.py
git commit -m "feat: add explicit embedding backend routing modes"
```

### Task 3: Add dedicated CPU embeddings service

**Files:**
- Create: `model-serving/embeddings-cpu/app.py`
- Create: `model-serving/embeddings-cpu/requirements.txt`
- Create: `model-serving/embeddings-cpu/Dockerfile`
- Modify: `model-serving/docker-compose.yml`
- Modify: `model-serving/.env.example`

- [ ] **Step 1: Implement CPU embeddings OpenAI-compatible endpoint**

```python
# model-serving/embeddings-cpu/app.py
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from sentence_transformers import SentenceTransformer

app = FastAPI(title='embeddings-cpu', version='0.1.0')
MODEL_NAME = os.getenv('EMBEDDING_CPU_MODEL_NAME', 'jinaai/jina-embeddings-v3')
API_KEY = os.getenv('EMBEDDING_API_KEY', 'local-dev-key')
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device='cpu')
    return _model


def check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing bearer token.')
    token = authorization.split(' ', 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid token.')


@app.get('/v1/models')
def models() -> dict[str, Any]:
    return {'object': 'list', 'data': [{'id': MODEL_NAME, 'object': 'model'}]}


@app.post('/v1/embeddings')
def embeddings(payload: dict, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    check_auth(authorization)
    if 'input' not in payload:
        raise HTTPException(status_code=400, detail='Request must include input.')
    inputs = payload['input'] if isinstance(payload['input'], list) else [payload['input']]
    vectors = get_model().encode(inputs, normalize_embeddings=True)
    data = [{'object': 'embedding', 'index': i, 'embedding': vectors[i].tolist()} for i in range(len(inputs))]
    return {'object': 'list', 'data': data, 'model': MODEL_NAME, 'usage': {'prompt_tokens': 0, 'total_tokens': 0}}
```

- [ ] **Step 2: Add CPU service dependencies and container build**

```txt
# model-serving/embeddings-cpu/requirements.txt
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.30.0,<1.0.0
sentence-transformers>=3.0.1,<4.0.0
```

```dockerfile
# model-serving/embeddings-cpu/Dockerfile
FROM python:3.11-slim
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Extend compose with mode-aware services**

```yaml
# model-serving/docker-compose.yml (excerpt)
services:
  embedding-vllm:
    profiles: ["vllm"]
    ...
  embeddings-cpu:
    profiles: ["cpu"]
    build:
      context: ./embeddings-cpu
      dockerfile: Dockerfile
    environment:
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY:-local-dev-key}
      - EMBEDDING_CPU_MODEL_NAME=${EMBEDDING_CPU_MODEL_NAME:-jinaai/jina-embeddings-v3}
    ports:
      - "18003:8000"
  model-gateway:
    environment:
      - EMBEDDING_BACKEND_MODE=${EMBEDDING_BACKEND_MODE:-vllm}
      - EMBEDDING_VLLM_BASE_URL=http://embedding-vllm:8000/v1
      - EMBEDDING_CPU_BASE_URL=http://embeddings-cpu:8000/v1
```

- [ ] **Step 4: Update env template for both modes**

```bash
# model-serving/.env.example (new/updated keys)
EMBEDDING_BACKEND_MODE=vllm
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_CPU_MODEL_NAME=jinaai/jina-embeddings-v3
VLLM_MAX_MODEL_LEN=512
VLLM_DTYPE=half
```

- [ ] **Step 5: Validate compose configuration**

Run:
```bash
cd model-serving && docker compose --profile vllm config >/tmp/compose-vllm.yaml && docker compose --profile cpu config >/tmp/compose-cpu.yaml && wc -l /tmp/compose-vllm.yaml /tmp/compose-cpu.yaml
```

Expected: both configs render successfully; non-zero line counts for both files.

- [ ] **Step 6: Commit CPU service wiring**

```bash
git add model-serving/embeddings-cpu model-serving/docker-compose.yml model-serving/.env.example
git commit -m "feat: add cpu embeddings service and compose profiles"
```

### Task 4: Make operations script mode-aware

**Files:**
- Modify: `scripts/model-serving.sh`

- [ ] **Step 1: Add mode parser and profile selection**

```bash
resolve_mode() {
  local env_file="$MS_DIR/.env"
  local mode="${EMBEDDING_BACKEND_MODE:-}"
  if [[ -z "$mode" && -f "$env_file" ]]; then
    mode="$(grep -E '^EMBEDDING_BACKEND_MODE=' "$env_file" | head -n1 | cut -d= -f2-)"
  fi
  [[ -z "$mode" ]] && mode="vllm"
  if [[ "$mode" != "vllm" && "$mode" != "transformers_cpu" ]]; then
    echo "Invalid EMBEDDING_BACKEND_MODE=$mode (use vllm or transformers_cpu)"
    exit 1
  fi
  echo "$mode"
}
```

- [ ] **Step 2: Use profile-aware compose commands**

```bash
start_stack() {
  cd "$MS_DIR"
  local mode profile
  mode="$(resolve_mode)"
  profile=$([[ "$mode" == "vllm" ]] && echo "vllm" || echo "cpu")
  $COMPOSE --profile "$profile" up -d --build
  $COMPOSE ps
}
```

- [ ] **Step 3: Split smoke checks by mode**

```bash
smoke_test() {
  local mode
  mode="$(resolve_mode)"
  wait_http_ok "gateway" "http://127.0.0.1:8002/readyz" 30 2
  if [[ "$mode" == "vllm" ]]; then
    wait_http_ok "vllm" "http://127.0.0.1:18001/v1/models" 120 3
  else
    wait_http_ok "embeddings-cpu" "http://127.0.0.1:18003/v1/models" 120 3
  fi
}
```

- [ ] **Step 4: Validate script syntax and help output**

Run:
```bash
bash -n scripts/model-serving.sh && ./scripts/model-serving.sh
```

Expected: no syntax error, usage text printed with exit code `1`.

- [ ] **Step 5: Commit script changes**

```bash
git add scripts/model-serving.sh
git commit -m "chore: make model-serving script mode-aware"
```

### Task 5: Docs, verification, and release-ready checks

**Files:**
- Modify: `model-serving/README.md`
- Modify: `docs/local-vllm.md`
- Create: `model-serving/.env.prod.vllm.example`
- Create: `model-serving/.env.dev.cpu.example`
- Test: `tests/model_serving/test_gateway.py`

- [ ] **Step 1: Document mode-based startup and API testing**

```markdown
## Backend modes
- `EMBEDDING_BACKEND_MODE=vllm`: vLLM only, fail fast on upstream errors.
- `EMBEDDING_BACKEND_MODE=transformers_cpu`: CPU embeddings service only.

### Start with vLLM mode
cp .env.prod.vllm.example .env
../scripts/model-serving.sh start

### Start with CPU mode
cp .env.dev.cpu.example .env
../scripts/model-serving.sh start
```

- [ ] **Step 2: Add explicit env templates**

```bash
# model-serving/.env.prod.vllm.example
EMBEDDING_BACKEND_MODE=vllm
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
VLLM_MAX_MODEL_LEN=512
VLLM_DTYPE=half
```

```bash
# model-serving/.env.dev.cpu.example
EMBEDDING_BACKEND_MODE=transformers_cpu
EMBEDDING_CPU_MODEL_NAME=jinaai/jina-embeddings-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

- [ ] **Step 3: Run full quality checks**

Run:
```bash
. .venv/bin/activate && ruff check . && pytest -q
```

Expected: lint passes and all tests pass.

- [ ] **Step 4: Run mode smoke tests**

Run:
```bash
cp model-serving/.env.prod.vllm.example model-serving/.env && ./scripts/model-serving.sh restart && ./scripts/model-serving.sh test
cp model-serving/.env.dev.cpu.example model-serving/.env && ./scripts/model-serving.sh restart && ./scripts/model-serving.sh test
```

Expected: smoke tests pass in both modes; gateway `/v1/embeddings` responds successfully in selected mode.

- [ ] **Step 5: Commit docs and templates**

```bash
git add model-serving/README.md docs/local-vllm.md model-serving/.env.prod.vllm.example model-serving/.env.dev.cpu.example
git commit -m "docs: add explicit embedding backend mode runbooks"
```

- [ ] **Step 6: Final release checkpoint commit**

```bash
git add -A
git commit -m "feat: add explicit vllm/cpu embedding backend routing"
```

