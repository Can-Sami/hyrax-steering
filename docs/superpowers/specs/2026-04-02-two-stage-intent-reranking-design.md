# Two-Stage Intent Retrieval Design (Semantic + Cross-Encoder Reranking)

## Problem
Current intent retrieval uses only vector semantic similarity. This can over-score intents that share banking nouns but differ in action verbs (for example, "pay bill" vs "check bill"), causing top-1 confusion.

## Objective
Add a new two-stage retrieval endpoint that:
- keeps legacy semantic search unchanged,
- retrieves top-K semantic candidates first,
- reranks those candidates against the user utterance with `BAAI/bge-reranker-v2-m3` through a vLLM scoring API,
- returns final ranking with both semantic and reranker scores.

## Scope
### In scope
- New endpoint for two-stage retrieval.
- New reranker provider/service for vLLM scoring API integration.
- New response contract including both score types.
- Explicit failure when reranker is unavailable/errors.
- Unit/API tests for ranking and failure behavior.
- API documentation update.

### Out of scope
- Replacing legacy `/v1/intents/search`.
- Changing confidence policy for inference endpoint in this iteration.
- ANN/index tuning or DB schema changes.

## API Design
### Existing endpoint (unchanged)
- `POST /api/v1/intents/search`
- Behavior: semantic-only retrieval.

### New endpoint
- `POST /api/v1/intents/search/rerank`
- Request:
  - `query: str` (required)
  - `k: int` (default 5, bounds same as legacy)
  - `language_hint: str` (default `tr`)
- Response:
```json
{
  "items": [
    {
      "intent_code": "bill_payment",
      "semantic_score": 0.86,
      "reranker_score": 0.42
    }
  ]
}
```
- Ordering: descending `reranker_score`.
- Candidate set: top-K intents from semantic stage only.

## Two-Stage Retrieval Flow
1. Clean and validate request (`query`, `k`, `language_hint`) using existing hygiene patterns.
2. Embed `query` with current embedding provider.
3. Retrieve semantic top-K via existing `SimilaritySearchService.top_k(...)`.
4. Build reranker pairs:
   - query = user utterance,
   - document = canonical intent description only (`Intent.description`).
5. Call vLLM scoring API with model `BAAI/bge-reranker-v2-m3`.
6. Combine semantic + reranker scores into result objects.
7. Sort by reranker score desc and return.

## Architecture Changes
### Domain model
- Add a new candidate shape that carries both scores (for new endpoint response mapping).
  - Example fields: `intent_id`, `intent_code`, `semantic_score`, `reranker_score`.

### Repository/service boundary
- Add a read helper to retrieve canonical intent descriptions by ids for stage-2 inputs.
  - This avoids changing semantic query path and keeps reranking data source explicit.

### Reranker provider
- Add `RerankerProvider` abstraction with concrete OpenAI-compatible implementation for vLLM scoring endpoint.
- Configurable via settings:
  - `reranker_engine` (default `openai_compatible`)
  - `reranker_model_name` (default `BAAI/bge-reranker-v2-m3`)
  - `reranker_base_url` (default to current OpenAI-compatible base)
  - `reranker_api_key` (default to existing API key)
- Request payload shape follows deployed vLLM scoring API contract for pairwise scoring.

### Orchestration service
- Add `TwoStageIntentSearchService`:
  - input: `query`, `embedding`, `k`, `language_code`
  - stage 1: semantic retrieval
  - stage 2: reranker scoring over canonical descriptions
  - output: ranked list with both scores

## Error Handling
- If reranker call fails (network, upstream error, invalid response), return explicit app error:
  - `code`: `reranker_engine_error`
  - `status_code`: `502`
  - message: clear upstream failure message consistent with existing provider errors.
- No fallback to semantic-only on this endpoint.
- Legacy semantic endpoint remains unaffected.

## Testing Strategy
### Unit tests
- Reranker provider:
  - successful parse of scoring response.
  - invalid payload / upstream failure maps to `AppError(reranker_engine_error, 502)`.
- Two-stage service:
  - preserves top-K candidate set from semantic stage.
  - reranks correctly by reranker score.
  - carries semantic score through to response object.

### API tests
- `POST /api/v1/intents/search/rerank` success:
  - response includes both `semantic_score` and `reranker_score`.
  - ordering uses reranker score.
- failure path:
  - reranker service error returns expected app error contract.
- ensure `/api/v1/intents/search` contract and behavior remain unchanged.

## Rollout Notes
- Backward compatibility is preserved by endpoint separation.
- Existing clients can stay on semantic-only search.
- New clients can opt into two-stage endpoint per use case.

## Open Integration Constraint
- This design assumes a reachable vLLM scoring API endpoint in runtime environments.
- If current model gateway does not expose scoring routes yet, a follow-up change is required in `model-serving/gateway` to proxy the scoring endpoint; backend contract here remains stable once endpoint is available.
