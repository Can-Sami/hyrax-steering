# Implementation Plan — Intent Recognition Backend (Turkish-first, OpenShift-ready)

## Objective
Deliver a production-grade backend that accepts IVR audio, performs STT + embedding-based similarity search, and returns intent classification with confidence, while supporting both CPU and GPU runtime profiles.

## Execution Mode
- **Development mode:** local-first
- **Deployment target:** OpenShift
- **Runtime profiles:** `cpu` and `gpu` (same API contract)
- **Language runtime in V1:** Turkish (`tr`) only

## Workstreams
- **A. Platform foundation**
- **B. Data model + vector retrieval**
- **C. Inference pipeline (Whisper + Qwen embeddings)**
- **D. API and policy layer**
- **E. Evaluation and quality gates**
- **F. OpenShift packaging and operations**

## Dependency Graph
- Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 → Step 8 → Step 9 → Step 10
- Parallelism:
  - Step 6 (intent admin APIs) and Step 7 (inference API) can run in parallel after Step 5.
  - Step 9 (OpenShift manifests) can start once Step 5 stabilizes, then refine after Step 7.

---

## Step 1 — Bootstrap project skeleton
**Goal:** Create backend scaffolding with clean module boundaries for API, domain, infra, and workers.

**Scope**
- Initialize Python project (FastAPI, settings, logging).
- Create package layout:
  - `app/api`, `app/domain`, `app/services`, `app/workers`, `app/db`, `app/config`.
- Add health and readiness endpoints.
- Add base error handling (explicit, typed errors).

**Deliverables**
- App starts locally.
- Config system supports env-based overrides.
- Structured JSON logging with request ID propagation.

**Verification**
- `uv run ruff check .`
- `uv run pytest -q`
- `uv run python -m app.main` starts successfully.

**Exit criteria**
- Baseline service boots and exposes `/healthz` and `/readyz`.

---

## Step 2 — Database and migration baseline (PostgreSQL + pgvector)
**Goal:** Stand up schema and migration flow for intents, embeddings, and inference records.

**Scope**
- Add migration tool (Alembic).
- Enable pgvector extension migration.
- Create V1 tables:
  - `intents`, `intent_utterances`, `intent_embeddings`, `inference_requests`, `inference_results`, `model_registry`.
- Add indexes (intent lookup, language lookup, vector similarity).

**Deliverables**
- Reproducible DB schema migration pipeline.
- Seed script for minimal Turkish intents dataset.

**Verification**
- `uv run alembic upgrade head`
- `uv run pytest -q tests/integration/test_schema.py`

**Exit criteria**
- New DB instance can be fully created from migrations with no manual SQL steps.

---

## Step 3 — Intent retrieval repository + similarity engine
**Goal:** Implement reliable top-k vector retrieval in application code.

**Scope**
- Build repository layer for intent/utterance CRUD.
- Implement vector search query abstraction for pgvector distance metrics.
- Return deterministic top-k candidate list.
- Add score normalization and confidence calculation primitives.

**Deliverables**
- `IntentRepository`, `EmbeddingRepository`, `SimilaritySearchService`.
- Unit tests for thresholding and ranking behavior.

**Verification**
- `uv run pytest -q tests/unit/test_similarity_search.py`
- `uv run pytest -q tests/integration/test_pgvector_search.py`

**Exit criteria**
- For seeded examples, expected intent appears in top-k with stable ordering.

---

## Step 4 — Audio ingestion + storage abstraction
**Goal:** Securely ingest IVR audio and route it to processing pipeline.

**Scope**
- Implement upload validator (codec, size, max duration).
- Add storage provider interface:
  - local filesystem provider for dev
  - S3-compatible provider for OpenShift
- Persist request metadata and audio URI in `inference_requests`.

**Deliverables**
- Ingestion service with strict input validation.
- Idempotency behavior keyed by `external_request_id`.

**Verification**
- `uv run pytest -q tests/unit/test_audio_validation.py`
- `uv run pytest -q tests/integration/test_ingest_flow.py`

**Exit criteria**
- Invalid audio is rejected with explicit error contract; valid audio is stored and tracked.

---

## Step 5 — Inference worker pipeline (CPU/GPU selectable)
**Goal:** Implement STT + embedding worker with first-class CPU support.

**Scope**
- Worker queue and job contract.
- Whisper transcription stage (`whisper-large-v3`) with Turkish default.
- Embedding stage (`qwen3-VL-Embedding-8B`).
- Runtime device mode:
  - `INFERENCE_DEVICE=cpu|cuda|auto`
  - `MODEL_PRECISION=fp32|fp16|int8` with safe compatibility checks.
- Model warmup, timeout handling, and explicit failure states.

**Deliverables**
- `SttProvider`, `EmbeddingProvider`, `InferencePipeline`.
- CPU profile configs + GPU profile configs.

**Verification**
- `uv run pytest -q tests/unit/test_device_selection.py`
- `uv run pytest -q tests/integration/test_worker_pipeline.py`

**Exit criteria**
- Same job succeeds in both CPU and GPU profile environments (where hardware exists).

---

## Step 6 — Intent management APIs (parallel lane)
**Goal:** Build admin APIs for intent and utterance lifecycle.

**Scope**
- `GET /v1/intents`
- `POST /v1/intents`
- `POST /v1/intents/reindex`
- Validation for language metadata (Turkish active, multilingual schema-ready).

**Deliverables**
- API endpoints + service layer.
- Reindex job trigger flow.

**Verification**
- `uv run pytest -q tests/api/test_intents_api.py`

**Exit criteria**
- Intent catalog can be created, listed, and reindexed through API only.

---

## Step 7 — Inference API contract (parallel lane)
**Goal:** Expose caller-facing inference endpoint for IVR integration.

**Scope**
- Implement `POST /v1/inference/intent` multipart contract.
- Support sync response for short audio and async fallback for longer workloads.
- Response fields:
  - `intent_code`, `confidence`, `match_status`, `transcript`, `top_candidates`, `processing_ms`.
- Include `low_confidence` behavior.

**Deliverables**
- Stable external API contract with OpenAPI examples.
- Error model (validation, inference, policy, transient infra errors).

**Verification**
- `uv run pytest -q tests/api/test_inference_api.py`
- `uv run pytest -q tests/contract/test_inference_contract.py`

**Exit criteria**
- IVR can send audio and receive deterministic intent classification payload.

---

## Step 8 — Confidence policy + evaluation harness
**Goal:** Define measurable quality gates before production.

**Scope**
- Add offline evaluation dataset tooling for Turkish intents.
- Implement policy tuning:
  - global threshold baseline
  - optional per-intent override support
- Generate metrics report:
  - top-1 accuracy, top-k recall, low-confidence precision, confusion matrix.

**Deliverables**
- Evaluation scripts and report template.
- Versioned policy config (`policy_version` persisted in results).

**Verification**
- `uv run python -m app.eval.run --dataset data/eval/tr_v1.jsonl`
- `uv run pytest -q tests/eval`

**Exit criteria**
- A documented threshold policy is chosen with reproducible metrics output.

---

## Step 9 — OpenShift deployment assets (cpu/gpu overlays)
**Goal:** Prepare production deployment manifests with profile isolation.

**Scope**
- Create `deploy/openshift/base`.
- Create overlays:
  - `deploy/openshift/cpu`
  - `deploy/openshift/gpu`
- Add:
  - Deployments, Services, HPAs/KEDA, ConfigMaps, Secrets references, probes.
  - Worker and API resource requests/limits per profile.
  - Optional node selectors/tolerations for GPU.

**Deliverables**
- Repeatable deployment manifests for both runtime profiles.
- Runbook for profile switching.

**Verification**
- `oc kustomize deploy/openshift/cpu`
- `oc kustomize deploy/openshift/gpu`

**Exit criteria**
- Both overlays render valid manifests with no contract drift.

---

## Step 10 — Production hardening + release readiness
**Goal:** Ensure reliability, security, and operability before first release.

**Scope**
- Add auth strategy (API key/JWT) for IVR clients.
- Add observability:
  - metrics (latency, queue depth, confidence bands)
  - structured logs
  - trace correlation IDs
- Add CI workflow for lint, test, migration checks, image build.
- Write operational docs:
  - local dev runbook
  - OpenShift deploy runbook
  - CPU/GPU switch guide
  - incident checklist.

**Deliverables**
- Release checklist with pass/fail gates.
- Baseline SLO targets for inference latency and error rate.

**Verification**
- `uv run ruff check . && uv run pytest -q`
- CI pipeline green on main branch.

**Exit criteria**
- System is deployable and supportable in OpenShift with defined quality gates.

---

## Adversarial Review Checklist (must pass before implementation starts)
- No step depends on hidden context from prior sessions.
- CPU-mode viability validated in at least one explicit test path.
- API contracts remain identical across cpu/gpu profiles.
- Schema supports multilingual expansion without breaking Turkish V1.
- Failure states are explicit (no silent fallback, no broad catches).
- Security controls exist for upload, auth, and logging hygiene.

## Mutation Protocol (if requirements change)
- **Split step:** when exit criteria become too large.
- **Insert step:** when a new mandatory dependency appears.
- **Reorder step:** only if dependency graph still validates.
- **Skip step:** only with documented risk and compensating control.
- **Abort track:** only if a replacement design is approved and recorded.

## Suggested First Sprint (Execution Start)
1. Step 1 (skeleton)
2. Step 2 (schema + migrations)
3. Step 3 (vector retrieval core)
4. Step 4 (ingestion)

This sequence creates the minimum stable backbone before model inference integration.
