# API Endpoints Quick Guide

Base URL: `http://localhost:18000`  
API prefix: `/api`  
Content type: `application/json` unless noted.

## Health

### `GET /api/healthz`
Checks service liveness.

Example:
```bash
curl -s http://localhost:18000/api/healthz
```

Response:
```json
{
  "status": "ok",
  "request_id": "..."
}
```

### `GET /api/readyz`
Checks readiness + version metadata.

```bash
curl -s http://localhost:18000/api/readyz
```

```json
{
  "status": "ready",
  "service": "callsteering-backend",
  "version": "0.1.0",
  "request_id": "..."
}
```

## Intent Management

### `GET /api/v1/intents`
Lists intents.

```bash
curl -s http://localhost:18000/api/v1/intents
```

```json
{
  "items": [
    {
      "id": "uuid",
      "intent_code": "balance_inquiry",
      "description": "Customer asks account balance",
      "is_active": true
    }
  ]
}
```

### `POST /api/v1/intents`
Creates an intent, adds canonical utterance from description, and stores embedding.

```bash
curl -s -X POST http://localhost:18000/api/v1/intents \
  -H "Content-Type: application/json" \
  -d '{"intent_code":"balance_inquiry","description":"Customer asks account balance"}'
```

```json
{
  "id": "uuid",
  "intent_code": "balance_inquiry",
  "created_at": "2026-03-31T23:59:59.000000+00:00"
}
```

### `PUT /api/v1/intents/{intent_id}`
Updates an intent and refreshes canonical utterance embedding.

```bash
curl -s -X PUT http://localhost:18000/api/v1/intents/<intent_id> \
  -H "Content-Type: application/json" \
  -d '{"intent_code":"balance_inquiry","description":"Updated description"}'
```

```json
{
  "id": "uuid",
  "intent_code": "balance_inquiry"
}
```

### `DELETE /api/v1/intents/{intent_id}`
Deletes an intent.

```bash
curl -s -X DELETE http://localhost:18000/api/v1/intents/<intent_id>
```

```json
{
  "status": "deleted",
  "id": "uuid"
}
```

### `POST /api/v1/intents/reindex`
Rebuilds embeddings for all utterances of active intents.

```bash
curl -s -X POST http://localhost:18000/api/v1/intents/reindex
```

```json
{
  "status": "ok",
  "reindexed_count": 34
}
```

### `GET /api/v1/intents/{intent_id}/utterances`
Lists utterances for an intent cluster.

```bash
curl -s http://localhost:18000/api/v1/intents/<intent_id>/utterances
```

```json
{
  "items": [
    {
      "id": "uuid",
      "intent_id": "uuid",
      "language_code": "tr",
      "text": "Kart borcumu ogrenmek istiyorum",
      "source": "manual"
    }
  ]
}
```

### `POST /api/v1/intents/{intent_id}/utterances`
Adds a new utterance to an intent cluster and stores embedding.

```bash
curl -s -X POST http://localhost:18000/api/v1/intents/<intent_id>/utterances \
  -H "Content-Type: application/json" \
  -d '{"text":"Kart borcumu ogrenmek istiyorum","language_code":"tr","source":"manual"}'
```

```json
{
  "id": "uuid",
  "intent_id": "uuid"
}
```

### `PUT /api/v1/intents/{intent_id}/utterances/{utterance_id}`
Updates utterance text/metadata and refreshes embedding.

```bash
curl -s -X PUT http://localhost:18000/api/v1/intents/<intent_id>/utterances/<utterance_id> \
  -H "Content-Type: application/json" \
  -d '{"text":"Kart borcum ne kadar?","language_code":"tr","source":"manual"}'
```

### `DELETE /api/v1/intents/{intent_id}/utterances/{utterance_id}`
Deletes an utterance from an intent cluster.

```bash
curl -s -X DELETE http://localhost:18000/api/v1/intents/<intent_id>/utterances/<utterance_id>
```

### `POST /api/v1/intents/search`
Semantic search by query text across utterance embeddings; results are aggregated by intent with max similarity.

```bash
curl -s -X POST http://localhost:18000/api/v1/intents/search \
  -H "Content-Type: application/json" \
  -d '{"query":"hesabimda ne kadar para var","k":5,"language_hint":"tr"}'
```

```json
{
  "items": [
    { "intent_code": "balance_inquiry", "score": 0.93 },
    { "intent_code": "card_limit", "score": 0.71 }
  ]
}
```

### `POST /api/v1/intents/search/rerank`
Two-stage search: semantic top-K retrieval followed by cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) via configured vLLM scoring API.

```bash
curl -s -X POST http://localhost:18000/api/v1/intents/search/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"faturami odemek istiyorum","k":5,"language_hint":"tr"}'
```

```json
{
  "items": [
    { "intent_code": "bill_payment", "semantic_score": 0.86, "reranker_score": 0.42 },
    { "intent_code": "bill_check", "semantic_score": 0.88, "reranker_score": 0.15 }
  ]
}
```

If reranker upstream fails, this endpoint returns:

```json
{
  "error": {
    "code": "reranker_engine_error",
    "message": "Failed to rerank candidates from configured engine."
  }
}
```

## Inference

### `POST /api/v1/inference/intent` (multipart/form-data)
Infers intent from uploaded audio file and persists request/result records server-side for analytics aggregation.

Auth: if `API_KEY` is configured, send header `x-api-key: <key>`.

```bash
curl -s -X POST http://localhost:18000/api/v1/inference/intent \
  -H "x-api-key: local-dev-key" \
  -F "audio_file=@/absolute/path/sample.wav" \
  -F "language_hint=tr" \
  -F "channel_id=ivr-01" \
  -F "request_id=req-123"
```

```json
{
  "request_id": "req-123",
  "channel_id": "ivr-01",
  "intent_code": "balance_inquiry",
  "confidence": 0.89,
  "match_status": "matched",
  "transcript": "hesabimda ne kadar para var",
  "top_candidates": [
    { "intent_code": "balance_inquiry", "score": 0.89 },
    { "intent_code": "card_limit", "score": 0.67 }
  ],
  "processing_ms": 120
}
```

## Overview Dashboard

Overview metrics are aggregated server-side from persisted inference data (`inference_requests` + `inference_results`) and filtered by explicit timeframe.

### Time filter parameters

- `start_at` (required): ISO timestamp (for example `2026-04-01T10:00:00Z`)
- `end_at` (required): ISO timestamp
- Rule: `start_at < end_at`

### `GET /api/v1/overview/summary`
Returns total inferences, average confidence, and previous-window delta percentages.

```bash
curl -s "http://localhost:18000/api/v1/overview/summary?start_at=2026-04-01T10:00:00Z&end_at=2026-04-01T12:00:00Z"
```

```json
{
  "total_inferences": 128,
  "avg_confidence": 0.78,
  "total_inferences_delta_pct": 14.29,
  "avg_confidence_delta_pct": -2.5
}
```

### `GET /api/v1/overview/intent-distribution`
Returns intent distribution for the window, including unmatched inferences.

```bash
curl -s "http://localhost:18000/api/v1/overview/intent-distribution?start_at=2026-04-01T10:00:00Z&end_at=2026-04-01T12:00:00Z"
```

```json
{
  "items": [
    { "intent_code": "balance_inquiry", "count": 74, "percentage": 57.81 },
    { "intent_code": "card_limit", "count": 30, "percentage": 23.44 },
    { "intent_code": "unmatched", "count": 24, "percentage": 18.75 }
  ]
}
```

### `GET /api/v1/overview/recent-activity`
Returns most recent inference records first.

Query params:
- `start_at` (required)
- `end_at` (required)
- `limit` (optional, default `10`, min `1`, max `50`)

```bash
curl -s "http://localhost:18000/api/v1/overview/recent-activity?start_at=2026-04-01T10:00:00Z&end_at=2026-04-01T12:00:00Z&limit=2"
```

```json
{
  "items": [
    {
      "timestamp": "2026-04-01T11:59:10+00:00",
      "input_snippet": "hesabimda ne kadar para var",
      "predicted_intent": "balance_inquiry",
      "confidence": 0.89
    },
    {
      "timestamp": "2026-04-01T11:58:31+00:00",
      "input_snippet": "temsilciye bagla",
      "predicted_intent": null,
      "confidence": 0.42
    }
  ]
}
```

## Error format

Most app-level errors return:

```json
{
  "error": {
    "code": "intent_not_found",
    "message": "Intent not found."
  }
}
```
