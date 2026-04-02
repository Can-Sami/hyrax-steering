# Utterance-Cluster Intent Retrieval Design

## Problem
Current retrieval uses one embedding per intent (`intent_code + description`). This under-represents customer phrasing variety and hurts top-1 classification when queries differ from the canonical description.

## Objective
Improve top-1 intent accuracy by modeling each intent as a cluster of utterances and ranking intents from utterance-level semantic matches.

## Decisions
- Keep `intent_code` as the class label.
- Store multiple utterances per intent.
- Store one embedding per utterance (not per intent).
- Retrieve top intent candidates by aggregating utterance similarities with `max(similarity)` per intent.
- Keep confidence policy unchanged for now.

## Data Model
- `intent_utterances`: unchanged role, canonical + example utterances.
- `intent_embeddings`: now linked to `utterance_id` (FK), unique on (`utterance_id`, `model_name`).

## API Changes
- Existing intent APIs remain.
- `POST /v1/intents` creates canonical utterance from `description`.
- `PUT /v1/intents/{id}` updates canonical utterance text and embedding.
- `POST /v1/intents/reindex` reindexes all utterances of active intents.
- New utterance APIs:
  - `GET /v1/intents/{id}/utterances`
  - `POST /v1/intents/{id}/utterances`
  - `PUT /v1/intents/{id}/utterances/{utterance_id}`
  - `DELETE /v1/intents/{id}/utterances/{utterance_id}`

## Retrieval Flow
1. Embed user query.
2. Compute cosine similarity against utterance embeddings filtered by language.
3. Group by intent and compute `max(score)` per intent.
4. Return top-K intents sorted by aggregated score.

## Error Handling
- Missing intent returns `intent_not_found`.
- Missing utterance under an intent returns `utterance_not_found`.
- Validation still rejects blank text/fields via existing request hygiene.

## Testing
- API tests cover utterance CRUD and intent CRUD/search compatibility.
- Schema migration tests verify `utterance_id` embedding linkage and uniqueness constraint.
- Existing inference contract remains stable.

## Scope Notes
- This change focuses on retrieval accuracy and intent admin ergonomics.
- No confidence-policy rewrite in this iteration.
- No ANN index tuning in this iteration.
