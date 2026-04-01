# Embedding Backend Routing Design (vLLM Primary + Transformers CPU Fallback Mode)

## Problem
The current model gateway serves embeddings through vLLM and transcriptions through Whisper. This works well for GPU environments, but local/dev and constrained environments may not have enough VRAM for target embedding models. We need a CPU-capable fallback path without weakening architecture quality or changing the backend integration contract.

## Goals
- Keep backend integration unchanged (`/v1/embeddings` OpenAI-compatible API).
- Preserve vLLM as the production-first backend.
- Support explicit CPU fallback mode for embeddings via transformers.
- Keep behavior deterministic (no silent auto-failover).
- Maintain testable, clean boundaries between routing, transport, and response normalization.

## Non-goals
- Dynamic runtime backend auto-selection.
- Transparent failover in `vllm` mode.
- Replacing Whisper ASR flow.

## Recommended approach
Use one gateway facade with explicit backend mode routing:
- `EMBEDDING_BACKEND_MODE=vllm`
- `EMBEDDING_BACKEND_MODE=transformers_cpu`

The gateway remains the single public endpoint and routes embeddings to one backend based on mode. vLLM remains the default deployment mode for production. CPU fallback is an intentional mode used where GPU capacity is unavailable or unsuitable.

## Alternatives considered
1. In-process transformers fallback inside gateway:
   - Pros: fewer containers.
   - Cons: larger gateway runtime surface, tighter coupling, harder resource isolation.
2. Automatic fallback from vLLM to CPU:
   - Pros: fewer outages in some scenarios.
   - Cons: hidden behavior shifts, unpredictable latency/cost profiles, ambiguous ops signals.
3. External infra routing outside gateway:
   - Pros: policy at platform layer.
   - Cons: more infra complexity, less app-level control and test coverage.
4. **Recommended** separate `embeddings-cpu` service behind gateway with explicit mode:
   - Clear boundaries, deterministic behavior, and production-safe vLLM alignment.

## Architecture
### Components
- `model-gateway` (public OpenAI-compatible facade):
  - Auth and request validation.
  - Embedding backend mode resolution.
  - Upstream request forwarding.
  - OpenAI response normalization.
- `embedding-vllm` (existing upstream in GPU environments).
- `embeddings-cpu` (new upstream service using transformers/sentence-transformers on CPU).

### Routing contract
- In `vllm` mode: gateway calls only `embedding-vllm`.
- In `transformers_cpu` mode: gateway calls only `embeddings-cpu`.
- No cross-mode fallback.

### API contract
Gateway keeps returning OpenAI-compatible embeddings response format in both modes so backend clients do not change.

## Data flow
1. Client calls `POST /v1/embeddings` on gateway.
2. Gateway validates API key and input schema.
3. Gateway resolves backend mode from environment.
4. Gateway forwards request to selected upstream.
5. Gateway normalizes response shape and metadata.
6. Gateway returns final response.

## Error handling
- Invalid mode/model input: `400`/`422` with clear message.
- Upstream unavailable/timeout in selected mode: `503`.
- Upstream client errors (`4xx`): propagate with context.
- Upstream server errors (`5xx`): map to gateway error with correlation id.
- In `vllm` mode, vLLM-down must not redirect to CPU backend.

## Observability and readiness
- Keep `/healthz` for process liveness.
- Add mode-aware readiness checks:
  - `/readyz?backend=vllm`
  - `/readyz?backend=transformers_cpu`
- Log selected backend per request and upstream latency.

## Configuration
- `EMBEDDING_BACKEND_MODE` enum: `vllm|transformers_cpu`.
- Existing embedding model vars remain backend-specific.
- Introduce backend-specific endpoints/config for CPU service.
- Provide environment templates:
  - `.env.prod.vllm`
  - `.env.dev.cpu`

## Testing strategy
- Unit tests:
  - mode resolution and validation
  - strict no-fallback behavior
  - response normalization consistency
- Integration tests:
  - vLLM mode success path
  - CPU mode success path
  - vLLM mode upstream down returns `503`
  - invalid mode/config handling
- Script-level smoke tests:
  - mode-aware startup checks
  - endpoint probes for selected mode

## Operational updates
- Extend compose profiles with optional `embeddings-cpu` service.
- Update `scripts/model-serving.sh`:
  - mode-aware `start/status/logs/test`
  - config sanity validation for selected mode
- Keep existing auth/security guards unchanged.

## Rollout plan
1. Introduce router abstraction and CPU upstream adapter.
2. Add CPU service and compose/profile wiring.
3. Add/extend tests for both modes and failure semantics.
4. Update runbooks and scripts.
5. Default production templates to `vllm`.

## Success criteria
- Backend continues to call one gateway base URL unchanged.
- In production profile, embeddings run via vLLM only.
- In CPU profile, embeddings run through transformers service reliably.
- Error semantics are explicit and deterministic.
- Test suite covers mode behavior and no-fallback guarantees.
