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
Creates an intent and stores embedding.

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
Updates an intent and refreshes embedding.

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
Rebuilds embeddings for all active intents.

```bash
curl -s -X POST http://localhost:18000/api/v1/intents/reindex
```

```json
{
  "status": "ok",
  "reindexed_count": 12
}
```

### `POST /api/v1/intents/search`
Semantic search by query text.

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

## Inference

### `POST /api/v1/inference/intent` (multipart/form-data)
Infers intent from uploaded audio file.

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
