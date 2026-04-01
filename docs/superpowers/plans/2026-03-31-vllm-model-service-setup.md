# vLLM Model Service Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and integrate a dedicated model-serving service that exposes OpenAI-compatible transcription and embedding endpoints for the backend.

**Architecture:** Add a separate `model-serving` project with two runtime components: a vLLM embedding server and a FastAPI gateway that serves Whisper ASR while proxying embeddings to vLLM. Keep backend integration unchanged by pointing its existing OpenAI-compatible providers to a single gateway base URL. Include OpenShift manifests and local run documentation.

**Tech Stack:** FastAPI, httpx, faster-whisper, vLLM OpenAI server, Docker Compose, OpenShift Kustomize, pytest, ruff.

---

### Task 1: Baseline verification and integration boundary lock

**Files:**
- Modify: `docs/local-vllm.md`
- Test: `tests/unit/test_provider_selection.py`

- [ ] **Step 1: Run baseline quality checks**

```bash
. .venv/bin/activate
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Expected: lint passes and tests pass.

- [ ] **Step 2: Confirm provider integration point with failing safety test**

```python
# tests/unit/test_storage.py
import pytest

from app.api.errors import AppError
from app.services.storage import LocalStorageProvider


def test_store_rejects_path_traversal_filename(tmp_path) -> None:
    provider = LocalStorageProvider(root_dir=str(tmp_path))
    with pytest.raises(AppError):
        provider.store('../evil.wav', b'abc')
```

- [ ] **Step 3: Run single test to verify failure before fix**

Run: `.venv/bin/pytest tests/unit/test_storage.py::test_store_rejects_path_traversal_filename -q`
Expected: FAIL before storage hardening exists.

- [ ] **Step 4: Commit baseline and failing-test checkpoint**

```bash
git add tests/unit/test_storage.py docs/local-vllm.md
git commit -m "test: capture storage filename traversal risk"
```

### Task 2: Scaffold model-serving project

**Files:**
- Create: `model-serving/gateway/app.py`
- Create: `model-serving/gateway/requirements.txt`
- Create: `model-serving/gateway/Dockerfile`
- Create: `model-serving/docker-compose.yml`
- Create: `model-serving/.env.example`
- Create: `model-serving/README.md`

- [ ] **Step 1: Add dependency file for gateway**

```txt
# model-serving/gateway/requirements.txt
fastapi>=0.115.0,<1.0.0
httpx>=0.27.0,<1.0.0
python-multipart>=0.0.9,<1.0.0
uvicorn[standard]>=0.30.0,<1.0.0
faster-whisper>=1.0.3,<2.0.0
```

- [ ] **Step 2: Implement gateway endpoints**

```python
# model-serving/gateway/app.py (core contract)
@app.post('/v1/embeddings')
async def embeddings(payload: dict, authorization: str | None = Header(default=None)) -> JSONResponse:
    check_auth(authorization)
    if 'input' not in payload:
        raise HTTPException(status_code=400, detail='Request must include input.')
    ...

@app.post('/v1/audio/transcriptions')
async def transcriptions(...):
    check_auth(authorization)
    ...
    return {'text': text}
```

- [ ] **Step 3: Add gateway container image build**

```dockerfile
# model-serving/gateway/Dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Add compose runtime for vLLM + gateway**

```yaml
# model-serving/docker-compose.yml
services:
  embedding-vllm:
    image: vllm/vllm-openai:latest
    command: ["--model", "Qwen/Qwen3-Embedding-8B", "--served-model-name", "Qwen/Qwen3-Embedding-8B", "--port", "8000"]
  model-gateway:
    build:
      context: ./gateway
```

- [ ] **Step 5: Add env template and quickstart docs**

```bash
# model-serving/.env.example
GATEWAY_API_KEY=local-dev-key
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

- [ ] **Step 6: Commit scaffold**

```bash
git add model-serving/
git commit -m "feat: scaffold model-serving gateway with vLLM embedding backend"
```

### Task 3: Security and robustness hardening

**Files:**
- Modify: `app/services/storage.py`
- Modify: `model-serving/gateway/app.py`
- Test: `tests/unit/test_storage.py`

- [ ] **Step 1: Implement filename sanitization in storage**

```python
safe_name = Path(filename).name
if safe_name != filename:
    raise AppError(code='invalid_audio_filename', message='Invalid audio filename.', status_code=400)
target = self.root / f'{uuid4()}-{safe_name}'
```

- [ ] **Step 2: Add production key guard and request validation in gateway**

```python
@app.on_event('startup')
def validate_security_config() -> None:
    if APP_ENV.lower() in {'prod', 'production'} and (is_default_key(GATEWAY_API_KEY) or is_default_key(EMBEDDING_API_KEY)):
        raise RuntimeError('Default API keys are not allowed in production mode.')
```

- [ ] **Step 3: Add passing regression tests**

```python
# tests/unit/test_storage.py
def test_store_writes_file_for_safe_name(tmp_path) -> None:
    provider = LocalStorageProvider(root_dir=str(tmp_path))
    stored = provider.store('safe.wav', b'abc')
    assert stored.endswith('safe.wav')
```

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/pytest tests/unit/test_storage.py -q`
Expected: PASS.

- [ ] **Step 5: Commit hardening**

```bash
git add app/services/storage.py model-serving/gateway/app.py tests/unit/test_storage.py
git commit -m "fix: harden storage and gateway auth validation"
```

### Task 4: OpenShift deployment wiring

**Files:**
- Modify: `deploy/openshift/base/kustomization.yaml`
- Create: `deploy/openshift/base/deployment-model-gateway.yaml`
- Create: `deploy/openshift/base/service-model-gateway.yaml`
- Modify: `deploy/openshift/base/deployment-api.yaml`
- Modify: `deploy/openshift/base/deployment-worker.yaml`
- Modify: `deploy/openshift/cpu/kustomization.yaml`
- Modify: `deploy/openshift/gpu/kustomization.yaml`

- [ ] **Step 1: Add model-gateway resources to base kustomization**

```yaml
resources:
  - deployment-api.yaml
  - deployment-worker.yaml
  - deployment-model-gateway.yaml
  - service.yaml
  - service-model-gateway.yaml
```

- [ ] **Step 2: Create model-gateway Deployment and Service**

```yaml
# deploy/openshift/base/deployment-model-gateway.yaml
kind: Deployment
metadata:
  name: callsteering-model-gateway
```

```yaml
# deploy/openshift/base/service-model-gateway.yaml
kind: Service
metadata:
  name: callsteering-model-gateway
```

- [ ] **Step 3: Wire API and worker env vars to gateway base URL**

```yaml
- name: OPENAI_BASE_URL
  value: http://callsteering-model-gateway/v1
- name: OPENAI_API_KEY
  value: local-dev-key
```

- [ ] **Step 4: Add CPU/GPU overlay patches for gateway runtime profile**

```yaml
# cpu
- op: replace
  path: /spec/template/spec/containers/0/env/2/value
  value: cpu
```

```yaml
# gpu
- op: replace
  path: /spec/template/spec/containers/0/env/2/value
  value: cuda
```

- [ ] **Step 5: Commit deployment manifests**

```bash
git add deploy/openshift/
git commit -m "feat: add openshift manifests for model gateway service"
```

### Task 5: Docs, spec, and full verification

**Files:**
- Modify: `docs/local-vllm.md`
- Create: `docs/superpowers/specs/2026-03-31-vllm-model-service-design.md`
- Create: `docs/superpowers/plans/2026-03-31-vllm-model-service-setup.md`

- [ ] **Step 1: Update local runbook for gateway usage**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export WHISPER_MODEL_NAME=openai/whisper-large-v3
```

- [ ] **Step 2: Write the design spec**

```markdown
# vLLM Model Service Design (Whisper ASR + Qwen Embeddings)
```

- [ ] **Step 3: Run complete project checks**

```bash
. .venv/bin/activate
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Expected: all checks pass.

- [ ] **Step 4: Commit docs and verification updates**

```bash
git add docs/
git commit -m "docs: add vllm model-service design and setup guides"
```
