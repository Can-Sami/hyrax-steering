# Intent Recognition Backend Design (IVR Call Steering)

## Problem Statement
Build a backend-only application that receives caller audio from an IVR system, transcribes speech to text, computes embeddings, performs similarity-based intent recognition, and returns the detected intent (plus confidence metadata).  
V1 targets Turkish runtime support, while architecture must be multilingual-ready.  
The system must run locally first and be deployment-ready for OpenShift.  
The platform must support both GPU and CPU model execution modes.

## Scope (V1)

### In Scope
- API endpoint accepting audio input from IVR systems.
- Speech-to-text using `whisper-large-v3`.
- Embedding generation using `qwen3-VL-Embedding-8B`.
- Intent matching via PostgreSQL + pgvector similarity search.
- Turkish runtime support.
- Confidence scoring + explicit low-confidence response behavior.
- OpenShift-ready containerization and configuration.
- CPU-mode execution support (with configurable performance profile).

### Out of Scope (V1)
- IVR routing logic and downstream telephony orchestration.
- Frontend UI.
- Human annotation portal (can be added later).
- Fully active multilingual runtime behavior (prepared in architecture only).

## Architecture Recommendation
Use a **modular FastAPI monolith** with clear internal boundaries, plus asynchronous workers for heavy ML tasks.

Rationale:
- Best balance of delivery speed and production quality.
- Clean migration path to microservices later.
- Strong Python ecosystem support for STT/embedding inference.
- Straightforward OpenShift deployment and autoscaling strategy.

## High-Level Components
- **API Service (FastAPI)**  
  Handles request validation, job submission, status/result retrieval, health checks, and admin endpoints.

- **Inference Worker**  
  Runs STT + embedding pipeline; supports configurable `device=cpu|cuda` and model precision settings.

- **PostgreSQL + pgvector**  
  Stores intent catalog, language metadata, embeddings, similarity scores, and request/audit logs.

- **Object Storage (local dev + S3-compatible in cluster)**  
  Stores uploaded audio files and optional derived artifacts.

- **Config & Secrets Layer**  
  Environment-driven config for model paths, device policy, DB connection, thresholds, and language settings.

- **Observability Stack**  
  Structured logs, metrics (latency per stage, queue depth, confidence distribution), and trace IDs.

## Data Flow (V1)
1. IVR sends audio file to `/v1/inference/intent`.
2. API validates codec/length/size and creates request job.
3. Worker fetches audio and runs Whisper transcription (Turkish mode).
4. Worker computes transcript embedding with Qwen model.
5. Worker queries pgvector for nearest intent candidates.
6. Scoring policy computes final confidence + top-k alternatives.
7. API returns:
   - `intent_code`
   - `confidence`
   - `transcript`
   - `alternatives[]`
   - `status` (`matched` or `low_confidence`)

## API Contract (Draft)

### `POST /v1/inference/intent`
Multipart/form-data:
- `audio_file` (required)
- `channel_id` (optional metadata)
- `request_id` (optional caller-provided id)
- `language_hint` (optional, default `tr`)

Response:
- `request_id`
- `intent_code | null`
- `confidence` (0..1)
- `match_status` (`matched` | `low_confidence` | `error`)
- `transcript`
- `top_candidates`: `[{intent_code, score}]`
- `processing_ms`

### `GET /v1/intents`
List active intents and metadata.

### `POST /v1/intents`
Create/refresh intent definitions and canonical examples (tr-first, multilingual fields included).

### `POST /v1/intents/reindex`
Recompute and store embeddings for changed intent examples.

## Data Model (PostgreSQL + pgvector)
- `intents`
  - `id`, `intent_code` (unique), `domain`, `is_active`, timestamps
- `intent_utterances`
  - `id`, `intent_id`, `language_code`, `text`, `source`, timestamps
- `intent_embeddings`
  - `id`, `utterance_id`, `model_name`, `embedding vector`, `norm`, timestamps
- `inference_requests`
  - `id`, `external_request_id`, `language_code`, `audio_uri`, `transcript`, `status`, timings, timestamps
- `inference_results`
  - `id`, `request_id`, `predicted_intent_id`, `confidence`, `top_k_json`, `policy_version`, timestamps
- `model_registry`
  - `id`, `model_key`, `version`, `device_mode`, `quantization`, `is_active`, timestamps

## CPU/GPU Support Strategy (Required)
CPU support is a first-class architecture concern, not a fallback hack.

### Runtime Modes
- `INFERENCE_DEVICE=cpu|cuda|auto`
- `INFERENCE_MODE=latency|throughput`
- `MODEL_PRECISION=fp32|fp16|int8` (mode-dependent support)

### CPU Profile
- Smaller worker concurrency by default.
- Optional quantized model artifacts for CPU efficiency.
- Request timeouts and max audio duration guardrails tuned for CPU throughput.
- Independent HPA/replica profile for CPU-only OpenShift nodes.

### GPU Profile
- Higher throughput workers with CUDA-enabled images/node selectors.
- Separate deployment profile with model warmup and larger batch options.

### Deployment Policy
- Provide two OpenShift overlays/profiles:
  - `openshift/cpu`
  - `openshift/gpu`
- Same API contract and DB schema in both profiles.
- Feature flags control model loading strategy.

## Multilingual-Ready Design
- Persist language code in intents, utterances, and inference records.
- Keep model/provider abstraction by language (even if only Turkish active now).
- Add language routing interface:
  - `LanguagePipelineSelector.select(language_code) -> stt_config, embedding_config, threshold_policy`

## Quality & Evaluation Strategy
- Build offline evaluation dataset for Turkish intents (train/validation/test splits).
- Track:
  - Top-1 accuracy
  - Top-k recall
  - Low-confidence precision
  - Confusion matrix by intent/domain
- Introduce configurable thresholds:
  - global threshold
  - optional per-intent threshold

## OpenShift Readiness Plan
- Containerized services with non-root images.
- Liveness/readiness/startup probes.
- ConfigMaps/Secrets for environment configuration.
- PVC or object store integration for audio/model cache where required.
- Separate resource requests/limits for API vs worker.
- Optional KEDA/HPA based on queue depth + CPU/GPU utilization.
- CI pipeline ready to build and push images for local and OpenShift targets.

## Security & Reliability
- Request auth via API key/JWT between IVR and backend.
- File-type and duration validation before inference.
- PII-safe logging policy (mask sensitive identifiers).
- Idempotency via `external_request_id`.
- Explicit failure responses (no silent fallback).

## Implementation Phases
1. Foundation & project skeleton (FastAPI, settings, health, DB wiring).
2. Intent catalog schema + pgvector integration.
3. STT + embedding pipeline abstraction with CPU/GPU selectable backends.
4. Inference endpoint and scoring policy.
5. Evaluation harness and threshold calibration.
6. OpenShift manifests (cpu/gpu profiles) + production hardening.

## Key Decisions Captured
- Framework: **FastAPI** (recommended over Django for this API-first ML backend).
- Input contract: **audio upload** from IVR; backend performs STT + intent recognition.
- V1 language runtime: **Turkish only**, multilingual-ready schema and abstractions.
- Output contract: **intent recognition only** (routing remains external).
- Infrastructure requirement: **CPU support mandatory**, GPU optional and configurable.

## Open Questions (For Next Planning Step)
- Synchronous vs async API response contract for long audio (blocking vs job polling).
- Accepted audio codecs and max duration for V1 SLA.
- Exact confidence threshold policy rollout (single threshold first or per-intent baseline).
- Packaging strategy for `qwen3-VL-Embedding-8B` in CPU mode (quantization format choice).
