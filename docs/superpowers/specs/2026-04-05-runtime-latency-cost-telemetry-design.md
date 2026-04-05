# Runtime Latency & Cost Telemetry Design (Inference Pipeline)

## Problem
We need high-confidence visibility into computational load, latency, and estimated cost for each inference stage so benchmark runs can determine viability and operating cost before rollout decisions.

Current inference flow executes STT, embedding, retrieval, confidence evaluation, and persistence, but does not persist stage-level timing/cost telemetry in an analyzable format.

## Objective
Add runtime telemetry that captures per-stage latency and estimated cost, then expose rollups that make benchmark comparisons straightforward and auditable.

Primary KPI:
- P50/P95 latency per stage
- Total estimated cost per 1k requests

## Scope
### In scope
- Runtime API telemetry (not a separate benchmark runner yet).
- Stage-level timing and estimated cost persistence in PostgreSQL.
- Hybrid storage model: per-request stage events + hourly rollups.
- Cost calculation based on measured usage metadata and versioned pricing rows.
- Comparison-ready overview endpoints for benchmark window analysis.

### Out of scope
- Replacing model providers or retrieval logic.
- Changing inference matching policy/threshold logic.
- Introducing Prometheus/OpenTelemetry as the primary storage path in this iteration.

## Stage Breakdown (Approved)
Telemetry will capture:
1. `stt`
2. `embedding`
3. `vector_search`
4. `rerank` (when endpoint/flow uses reranking)
5. `confidence_policy`
6. `persistence`
7. `total`

## Architecture
1. Add a request-scoped telemetry collector to inference request handling.
2. Wrap each stage with explicit start/stop timing using monotonic clock.
3. Collect stage metadata:
   - provider
   - model_name
   - usage_json (tokens, audio seconds, candidate counts, vector dimension, etc.)
4. Persist:
   - per-request summary (total latency/cost/status)
   - per-stage metrics rows (duration, usage, estimated cost, status/error fields)
5. Keep inference business output unchanged; telemetry is additive.

## Data Model
### 1) `inference_stage_metrics`
Per-stage event table.

Suggested fields:
- `id` (uuid, pk)
- `request_id` (fk -> `inference_requests.id`, indexed)
- `stage_name` (text/varchar)
- `started_at` (timestamptz)
- `finished_at` (timestamptz)
- `duration_ms` (int)
- `provider` (varchar)
- `model_name` (varchar)
- `usage_json` (jsonb)
- `estimated_cost_usd` (numeric)
- `status` (`ok` | `error`)
- `error_code` (varchar, nullable)
- `error_message` (text, nullable)
- `created_at` (timestamptz)

Indexes:
- (`stage_name`, `created_at`)
- (`request_id`)
- (`status`, `created_at`)

### 2) `inference_cost_pricing`
Versioned pricing catalog for audit-safe cost recomputation.

Suggested fields:
- `id` (uuid, pk)
- `provider` (varchar)
- `model_name` (varchar)
- `unit_type` (`token` | `audio_second` | `request` | `candidate`)
- `unit_price_usd` (numeric)
- `effective_from` (timestamptz)
- `effective_to` (timestamptz, nullable)
- `created_at` (timestamptz)

Constraint:
- One active row per (`provider`, `model_name`, `unit_type`, timestamp).

### 3) `inference_metrics_rollup_hourly`
Pre-aggregated analytics table for fast dashboard and benchmark queries.

Suggested fields:
- `bucket_start` (timestamptz)
- `stage_name` (varchar)
- `request_count` (int)
- `error_count` (int)
- `p50_ms` (float)
- `p95_ms` (float)
- `avg_ms` (float)
- `total_estimated_cost_usd` (numeric)
- `cost_per_1k_requests` (numeric)
- `created_at` (timestamptz)

Primary key:
- (`bucket_start`, `stage_name`)

## Cost & Latency Calculation Rules
1. Duration is measured directly per stage from monotonic timer.
2. Cost is estimated only when usage metadata is present and matched with active pricing row.
3. If usage/pricing is missing, `estimated_cost_usd` remains null and reason is captured in `usage_json`; no silent fallback.
4. `total` stage cost/latency is derived from observed request-level measurements, not constants.

## Error Handling
1. Stage failure writes telemetry row with `status=error`, plus `error_code`/`error_message`.
2. Telemetry write failures should be surfaced clearly and follow existing app error patterns.
3. No silent suppression of invalid telemetry payloads.

## Data Flow
1. Request enters inference endpoint.
2. Telemetry context starts `total`.
3. Each stage records timing and usage metadata.
4. Inference result persists as today.
5. Telemetry summary + stage rows persist.
6. Scheduled rollup job aggregates hourly stats.

## API/Reporting Surfaces
Add overview capabilities to support benchmarking:
1. Stage latency stats endpoint (p50/p95/avg by stage and time window).
2. Stage cost stats endpoint (total cost + cost per 1k by stage and window).
3. Optional recent traces endpoint for high-latency outlier debugging.
4. Benchmark comparison endpoint/query contract comparing two windows:
   - baseline window
   - candidate window
   - per-stage delta for p95 and cost-per-1k

## Benchmark Viability Evaluation
Comparison output should support pass/fail policy checks against configured targets:
- `p95_ms <= latency_budget_ms`
- `cost_per_1k_requests <= cost_budget_usd`

This enables objective go/no-go decisions per deployment candidate.

## Testing Strategy
### Unit
- Telemetry collector stage timing lifecycle.
- Cost calculator pricing resolution and null-cost behavior.
- Rollup math utilities (p50/p95/avg/cost-per-1k).

### Integration
- Inference request persists all approved stage rows.
- Failure path persists stage error metadata.
- Rollup generation produces expected aggregates.

### API
- Stage latency/cost endpoints return correct schema and values.
- Benchmark comparison endpoint returns correct deltas between windows.

## Rollout Notes
1. Introduce schema via migrations first.
2. Enable telemetry write path behind a feature flag if desired for safe rollout.
3. Backfill is optional; benchmarking works immediately for new traffic once enabled.

## Open Decisions (Resolved in this design)
- Runtime telemetry storage: PostgreSQL tables (selected).
- Cost strategy: pricing catalog + measured usage metadata (selected).
- Persistence model: hybrid event + rollup (selected).
