# vLLM Model Service Design (Whisper ASR + Qwen Embeddings)

## Problem
The backend already supports OpenAI-compatible STT and embedding providers, but needs a dedicated model-serving service for production-style integration.

## Recommended approach
Use a **single gateway service** fronting two inference engines:
- `vLLM` for `Qwen/Qwen3-Embedding-8B` embeddings
- `faster-whisper` runtime for `openai/whisper-large-v3` transcription

This keeps one backend base URL while using the right engine per model.

## Alternatives considered
1. Single vLLM-only process for both models: not viable because Whisper transcription endpoint support is not native in vLLM.
2. Two independent services exposed to backend: works, but adds backend config complexity and duplicated auth/rate-limit concerns.
3. **Recommended** gateway pattern: clean OpenAI-compatible contract, minimal backend changes.

## Architecture
- Backend calls one base URL (`/v1/embeddings`, `/v1/audio/transcriptions`).
- Gateway validates bearer token.
- Embeddings endpoint proxies to vLLM upstream.
- Transcription endpoint runs Whisper locally and returns OpenAI-style `{"text": ...}`.

## Error handling
- Unauthorized: `401`.
- Upstream embedding errors: `502`.
- Unsupported Whisper model override: `400`.
- File processing issues: `500`.

## Deployment
- Local: `docker compose` with `embedding-vllm` + `model-gateway`.
- OpenShift: add `callsteering-model-gateway` deployment + service in base, override resources in cpu/gpu overlays.

## Testing
- Keep backend tests green.
- Add gateway smoke checks (`/healthz`, `/readyz`) and manual endpoint verification.

## Scope
In scope: scaffolding and integration-ready runtime.
Out of scope: autoscaling policies, quantization tuning, model sharding.
