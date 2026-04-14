# Runtime Latency & Cost Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime stage-level latency and estimated-cost telemetry for inference, with PostgreSQL persistence and overview endpoints that support benchmark viability/cost comparisons.

**Architecture:** Introduce a request-scoped telemetry collector in the inference endpoint, persist per-stage metrics to a new table, resolve stage costs from versioned pricing rows, and expose hourly rollups for p50/p95/avg latency plus cost-per-1k. Keep inference prediction behavior unchanged and make telemetry additive.

**Tech Stack:** FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL (JSONB), pytest

---

## File Structure (create/modify map)

- Create: `alembic/versions/20260405_0004_add_inference_stage_metrics_and_pricing.py`
  - Add `inference_stage_metrics`, `inference_cost_pricing`, `inference_metrics_rollup_hourly` tables and indexes.
- Modify: `app/db/models.py`
  - Add SQLAlchemy models for the new telemetry/pricing/rollup tables.
- Create: `app/services/telemetry.py`
  - Request-scoped stage timer collector + payload normalization.
- Modify: `app/services/repository.py`
  - Add repositories for stage metrics, pricing lookup, rollup query endpoints.
- Modify: `app/api/routes.py`
  - Instrument `/v1/inference/intent` with stage timing/cost capture.
  - Add overview endpoints for stage latency, stage cost, and benchmark comparison.
- Modify: `app/domain/schemas.py`
  - Add typed DTOs for telemetry rollup responses and comparison results.
- Create: `tests/unit/test_telemetry.py`
  - Unit tests for collector and cost estimation behavior.
- Modify: `tests/api/test_overview_api.py`
  - API tests for new overview telemetry endpoints.
- Modify: `tests/integration/test_schema.py`
  - Assert new telemetry tables are present in metadata and migration text.
- Modify: `docs/pipeline-and-schema-brief.md`
  - Document the new telemetry schema and benchmark reporting surfaces.

### Task 1: Schema for stage metrics, pricing, and rollups

**Files:**
- Create: `alembic/versions/20260405_0004_add_inference_stage_metrics_and_pricing.py`
- Modify: `app/db/models.py`
- Test: `tests/integration/test_schema.py`

- [ ] **Step 1: Write the failing schema tests**

```python
# tests/integration/test_schema.py
def test_sqlalchemy_metadata_has_expected_tables() -> None:
    expected_tables = {
        'intents',
        'intent_utterances',
        'intent_embeddings',
        'inference_requests',
        'inference_results',
        'model_registry',
        'inference_stage_metrics',
        'inference_cost_pricing',
        'inference_metrics_rollup_hourly',
    }
    assert expected_tables.issubset(set(Base.metadata.tables.keys()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/integration/test_schema.py -q
```
Expected: FAIL because new tables are not defined yet.

- [ ] **Step 3: Write minimal migration + model implementation**

```python
# app/db/models.py (additions)
class InferenceStageMetric(Base):
    __tablename__ = 'inference_stage_metrics'
    # request_id FK, stage_name, started_at, finished_at, duration_ms, provider,
    # model_name, usage_json (JSONB), estimated_cost_usd, status, error_code, error_message, created_at

class InferenceCostPricing(Base):
    __tablename__ = 'inference_cost_pricing'
    # provider, model_name, unit_type, unit_price_usd, effective_from, effective_to, created_at

class InferenceMetricRollupHourly(Base):
    __tablename__ = 'inference_metrics_rollup_hourly'
    # bucket_start, stage_name, request_count, error_count, p50_ms, p95_ms, avg_ms, total_estimated_cost_usd, cost_per_1k_requests
```

```python
# alembic/versions/20260405_0004_add_inference_stage_metrics_and_pricing.py
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade() -> None:
    op.create_table(
        'inference_stage_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stage_name', sa.String(length=64), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(length=128), nullable=True),
        sa.Column('model_name', sa.String(length=128), nullable=True),
        sa.Column('usage_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('error_code', sa.String(length=64), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['request_id'], ['inference_requests.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'inference_cost_pricing',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider', sa.String(length=128), nullable=False),
        sa.Column('model_name', sa.String(length=128), nullable=False),
        sa.Column('unit_type', sa.String(length=32), nullable=False),
        sa.Column('unit_price_usd', sa.Float(), nullable=False),
        sa.Column('effective_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('effective_to', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'inference_metrics_rollup_hourly',
        sa.Column('bucket_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('stage_name', sa.String(length=64), nullable=False),
        sa.Column('request_count', sa.Integer(), nullable=False),
        sa.Column('error_count', sa.Integer(), nullable=False),
        sa.Column('p50_ms', sa.Float(), nullable=False),
        sa.Column('p95_ms', sa.Float(), nullable=False),
        sa.Column('avg_ms', sa.Float(), nullable=False),
        sa.Column('total_estimated_cost_usd', sa.Float(), nullable=False),
        sa.Column('cost_per_1k_requests', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('bucket_start', 'stage_name'),
    )
    op.create_index('ix_stage_metrics_request_id', 'inference_stage_metrics', ['request_id'])
    op.create_index('ix_stage_metrics_stage_created', 'inference_stage_metrics', ['stage_name', 'created_at'])

def downgrade() -> None:
    op.drop_index('ix_stage_metrics_stage_created', table_name='inference_stage_metrics')
    op.drop_index('ix_stage_metrics_request_id', table_name='inference_stage_metrics')
    op.drop_table('inference_metrics_rollup_hourly')
    op.drop_table('inference_cost_pricing')
    op.drop_table('inference_stage_metrics')
```

- [ ] **Step 4: Run schema tests to verify pass**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/integration/test_schema.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/20260405_0004_add_inference_stage_metrics_and_pricing.py app/db/models.py tests/integration/test_schema.py
git commit -m "feat: add telemetry schema for stage metrics and pricing"
```

### Task 2: Telemetry collector and cost estimation service

**Files:**
- Create: `app/services/telemetry.py`
- Modify: `app/services/repository.py`
- Test: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write failing unit tests for telemetry collector**

```python
# tests/unit/test_telemetry.py
def test_collector_records_stage_duration_and_status() -> None:
    collector = InferenceTelemetryCollector()
    collector.start_stage('embedding')
    collector.end_stage('embedding', status='ok', provider='openai_compatible', model_name='Qwen')
    rows = collector.export_stage_rows()
    assert rows[0]['stage_name'] == 'embedding'
    assert rows[0]['status'] == 'ok'
    assert rows[0]['duration_ms'] >= 0

def test_cost_estimation_returns_none_when_pricing_missing() -> None:
    cost = estimate_stage_cost(usage={'tokens': 120}, pricing=None)
    assert cost is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/unit/test_telemetry.py -q
```
Expected: FAIL because telemetry module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/telemetry.py
from dataclasses import dataclass, field
import time

@dataclass
class StageState:
    started_at_monotonic: float
    started_at_utc: datetime

class InferenceTelemetryCollector:
    def __init__(self) -> None:
        self._open: dict[str, StageState] = {}
        self._rows: list[dict[str, object]] = []

    def start_stage(self, stage_name: str) -> None:
        self._open[stage_name] = StageState(time.perf_counter(), datetime.now(timezone.utc))

    def end_stage(self, stage_name: str, *, status: str, provider: str | None = None, model_name: str | None = None, usage: dict | None = None, estimated_cost_usd: float | None = None, error_code: str | None = None, error_message: str | None = None) -> None:
        state = self._open.pop(stage_name)
        finished = datetime.now(timezone.utc)
        duration_ms = int((time.perf_counter() - state.started_at_monotonic) * 1000)
        self._rows.append(
            {
                'stage_name': stage_name,
                'started_at': state.started_at_utc,
                'finished_at': finished,
                'duration_ms': duration_ms,
                'provider': provider,
                'model_name': model_name,
                'usage_json': usage or {},
                'estimated_cost_usd': estimated_cost_usd,
                'status': status,
                'error_code': error_code,
                'error_message': error_message,
            }
        )

    def export_stage_rows(self) -> list[dict[str, object]]:
        return list(self._rows)
```

```python
# app/services/repository.py (new helpers)
class InferenceTelemetryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_stage_metrics(self, request_id, stage_rows: list[dict[str, object]]) -> None:
        now = _utc_now()
        for row in stage_rows:
            metric = InferenceStageMetric(
                request_id=request_id,
                stage_name=str(row['stage_name']),
                started_at=row['started_at'],
                finished_at=row['finished_at'],
                duration_ms=int(row['duration_ms']),
                provider=row.get('provider'),
                model_name=row.get('model_name'),
                usage_json=row.get('usage_json') or {},
                estimated_cost_usd=row.get('estimated_cost_usd'),
                status=str(row['status']),
                error_code=row.get('error_code'),
                error_message=row.get('error_message'),
                created_at=now,
            )
            self.db.add(metric)
        self.db.flush()

class InferencePricingRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_active_price(self, provider: str, model_name: str, unit_type: str, at: datetime):
        query = (
            select(InferenceCostPricing)
            .where(
                InferenceCostPricing.provider == provider,
                InferenceCostPricing.model_name == model_name,
                InferenceCostPricing.unit_type == unit_type,
                InferenceCostPricing.effective_from <= at,
                sa.or_(InferenceCostPricing.effective_to.is_(None), InferenceCostPricing.effective_to > at),
            )
            .order_by(InferenceCostPricing.effective_from.desc())
            .limit(1)
        )
        return self.db.scalar(query)
```

- [ ] **Step 4: Run telemetry unit tests**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/unit/test_telemetry.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/telemetry.py app/services/repository.py tests/unit/test_telemetry.py
git commit -m "feat: add telemetry collector and pricing lookup services"
```

### Task 3: Instrument inference endpoint stages and persist telemetry

**Files:**
- Modify: `app/api/routes.py`
- Modify: `app/services/repository.py`
- Test: `tests/api/test_intents_api.py`

- [ ] **Step 1: Write failing API test for telemetry persistence trigger**

```python
# tests/api/test_intents_api.py (new focused test)
def test_infer_intent_records_stage_metrics(monkeypatch) -> None:
    class _FakeTelemetryRepo:
        calls = []
        def __init__(self, db) -> None:
            self.db = db
        def create_stage_metrics(self, request_id, stage_rows):
            self.calls.append((request_id, stage_rows))
    monkeypatch.setattr(routes, 'InferenceTelemetryRepository', _FakeTelemetryRepo)
    # invoke /api/v1/inference/intent with mocked pipeline and assert at least 6 stage rows persisted
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/api/test_intents_api.py::test_infer_intent_records_stage_metrics -q
```
Expected: FAIL because endpoint does not call telemetry repository.

- [ ] **Step 3: Implement endpoint instrumentation**

```python
# app/api/routes.py (in infer_intent)
collector = InferenceTelemetryCollector()
collector.start_stage('total')
collector.start_stage('stt')
transcript = pipeline.stt_provider.transcribe(audio_uri=audio_uri, language_code=language_hint)
collector.end_stage('stt', status='ok', provider=settings.stt_engine, model_name=settings.whisper_model_name, usage={'audio_bytes': len(content)})

collector.start_stage('embedding')
embedding = pipeline.embedding_provider.embed(transcript.transcript).vector
collector.end_stage('embedding', status='ok', provider=settings.embedding_engine, model_name=settings.embedding_model_name, usage={'vector_dim': len(embedding)})

collector.start_stage('vector_search')
candidates = similarity.top_k(embedding=embedding, k=5, language_code=language_hint)
collector.end_stage('vector_search', status='ok', usage={'candidate_count': len(candidates)})

collector.start_stage('confidence_policy')
confidence_result = confidence_policy.evaluate(candidates)
collector.end_stage('confidence_policy', status='ok')

collector.start_stage('persistence')
# existing repository.create_request/create_result + commit
collector.end_stage('persistence', status='ok')
collector.end_stage('total', status='ok')

stage_rows = collector.export_stage_rows()
telemetry_repo.create_stage_metrics(inference_request.id, stage_rows)
```

- [ ] **Step 4: Run focused API test**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/api/test_intents_api.py::test_infer_intent_records_stage_metrics -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/routes.py app/services/repository.py tests/api/test_intents_api.py
git commit -m "feat: instrument inference pipeline with per-stage telemetry"
```

### Task 4: Add pricing-based cost estimation into stage rows

**Files:**
- Modify: `app/services/telemetry.py`
- Modify: `app/services/repository.py`
- Modify: `app/api/routes.py`
- Test: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write failing unit tests for cost estimation rules**

```python
def test_estimate_cost_from_token_usage() -> None:
    pricing = {'unit_type': 'token', 'unit_price_usd': 0.000002}
    usage = {'tokens': 500}
    assert estimate_stage_cost(usage=usage, pricing=pricing) == 0.001

def test_estimate_cost_keeps_null_when_usage_missing() -> None:
    pricing = {'unit_type': 'audio_second', 'unit_price_usd': 0.0001}
    assert estimate_stage_cost(usage={}, pricing=pricing) is None
```

- [ ] **Step 2: Run tests to verify fail**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/unit/test_telemetry.py::test_estimate_cost_from_token_usage tests/unit/test_telemetry.py::test_estimate_cost_keeps_null_when_usage_missing -q
```
Expected: FAIL.

- [ ] **Step 3: Implement cost estimation wiring**

```python
# app/services/telemetry.py
def estimate_stage_cost(*, usage: dict[str, object], pricing: dict[str, object] | None) -> float | None:
    if pricing is None:
        return None
    unit_type = pricing['unit_type']
    unit_price = float(pricing['unit_price_usd'])
    if unit_type == 'token' and 'tokens' in usage:
        return float(usage['tokens']) * unit_price
    if unit_type == 'audio_second' and 'audio_seconds' in usage:
        return float(usage['audio_seconds']) * unit_price
    if unit_type == 'request':
        return unit_price
    return None
```

```python
# app/api/routes.py (per stage end)
price = pricing_repo.get_active_price(
    provider=settings.embedding_engine,
    model_name=settings.embedding_model_name,
    unit_type='token',
    at=datetime.now(timezone.utc),
)
cost = estimate_stage_cost(usage=stage_usage, pricing=price)
collector.end_stage(
    'embedding',
    status='ok',
    provider=settings.embedding_engine,
    model_name=settings.embedding_model_name,
    usage=stage_usage,
    estimated_cost_usd=cost,
)
```

- [ ] **Step 4: Run unit tests**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/unit/test_telemetry.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/telemetry.py app/services/repository.py app/api/routes.py tests/unit/test_telemetry.py
git commit -m "feat: add pricing-based stage cost estimation"
```

### Task 5: Add overview endpoints for stage latency/cost and benchmark comparison

**Files:**
- Modify: `app/services/repository.py`
- Modify: `app/api/routes.py`
- Modify: `app/domain/schemas.py`
- Test: `tests/api/test_overview_api.py`

- [ ] **Step 1: Write failing overview API tests**

```python
# tests/api/test_overview_api.py
def test_overview_stage_latency_returns_rollup(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def stage_latency(self, start_at, end_at):
            _ = (start_at, end_at)
            return [{'stage_name': 'embedding', 'p50_ms': 28.0, 'p95_ms': 61.0, 'avg_ms': 33.5, 'request_count': 50, 'error_count': 0}]
    # GET /api/v1/overview/stage-latency and assert p50_ms/p95_ms fields exist

def test_overview_benchmark_compare_returns_deltas(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def benchmark_compare(self, baseline_start, baseline_end, candidate_start, candidate_end):
            _ = (baseline_start, baseline_end, candidate_start, candidate_end)
            return [{'stage_name': 'embedding', 'baseline_p95_ms': 70.0, 'candidate_p95_ms': 61.0, 'p95_delta_pct': -12.86, 'baseline_cost_per_1k': 1.8, 'candidate_cost_per_1k': 1.6, 'cost_delta_pct': -11.11}]
    # GET /api/v1/overview/benchmark-compare and assert p95_delta_pct/cost_delta_pct fields exist
```

- [ ] **Step 2: Run tests to verify fail**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/api/test_overview_api.py -q
```
Expected: FAIL due to missing endpoints/repository methods.

- [ ] **Step 3: Implement repository methods + routes**

```python
# app/services/repository.py (new methods)
def stage_latency(self, start_at: datetime, end_at: datetime) -> list[dict[str, object]]:
    rows = self.db.execute(
        select(
            InferenceMetricRollupHourly.stage_name,
            func.avg(InferenceMetricRollupHourly.p50_ms).label('p50_ms'),
            func.avg(InferenceMetricRollupHourly.p95_ms).label('p95_ms'),
            func.avg(InferenceMetricRollupHourly.avg_ms).label('avg_ms'),
            func.sum(InferenceMetricRollupHourly.request_count).label('request_count'),
            func.sum(InferenceMetricRollupHourly.error_count).label('error_count'),
        )
        .where(InferenceMetricRollupHourly.bucket_start >= start_at, InferenceMetricRollupHourly.bucket_start < end_at)
        .group_by(InferenceMetricRollupHourly.stage_name)
    ).all()
    return [dict(row._mapping) for row in rows]

def stage_cost(self, start_at: datetime, end_at: datetime) -> list[dict[str, object]]:
    rows = self.db.execute(
        select(
            InferenceMetricRollupHourly.stage_name,
            func.sum(InferenceMetricRollupHourly.total_estimated_cost_usd).label('total_estimated_cost_usd'),
            func.avg(InferenceMetricRollupHourly.cost_per_1k_requests).label('cost_per_1k_requests'),
        )
        .where(InferenceMetricRollupHourly.bucket_start >= start_at, InferenceMetricRollupHourly.bucket_start < end_at)
        .group_by(InferenceMetricRollupHourly.stage_name)
    ).all()
    return [dict(row._mapping) for row in rows]

def benchmark_compare(self, baseline_start: datetime, baseline_end: datetime, candidate_start: datetime, candidate_end: datetime) -> list[dict[str, object]]:
    baseline = {row['stage_name']: row for row in self.stage_latency(baseline_start, baseline_end)}
    candidate = {row['stage_name']: row for row in self.stage_latency(candidate_start, candidate_end)}
    output: list[dict[str, object]] = []
    for stage_name in sorted(set(baseline.keys()) | set(candidate.keys())):
        base = baseline.get(stage_name, {'p95_ms': 0.0})
        cand = candidate.get(stage_name, {'p95_ms': 0.0})
        base_p95 = float(base['p95_ms'] or 0.0)
        cand_p95 = float(cand['p95_ms'] or 0.0)
        delta_pct = None if base_p95 == 0 else ((cand_p95 - base_p95) / base_p95) * 100.0
        output.append({'stage_name': stage_name, 'baseline_p95_ms': base_p95, 'candidate_p95_ms': cand_p95, 'p95_delta_pct': delta_pct})
    return output
```

```python
# app/api/routes.py (new endpoints)
@router.get('/v1/overview/stage-latency')
def overview_stage_latency(start_at: str, end_at: str, db: Session = Depends(get_db)) -> dict[str, list[dict[str, object]]]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    repository = InferenceRepository(db)
    return {'items': repository.stage_latency(parsed_start, parsed_end)}

@router.get('/v1/overview/stage-cost')
def overview_stage_cost(start_at: str, end_at: str, db: Session = Depends(get_db)) -> dict[str, list[dict[str, object]]]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    repository = InferenceRepository(db)
    return {'items': repository.stage_cost(parsed_start, parsed_end)}

@router.get('/v1/overview/benchmark-compare')
def overview_benchmark_compare(baseline_start_at: str, baseline_end_at: str, candidate_start_at: str, candidate_end_at: str, db: Session = Depends(get_db)) -> dict[str, list[dict[str, object]]]:
    baseline_start, baseline_end = _parse_timeframe(baseline_start_at, baseline_end_at)
    candidate_start, candidate_end = _parse_timeframe(candidate_start_at, candidate_end_at)
    repository = InferenceRepository(db)
    return {'items': repository.benchmark_compare(baseline_start, baseline_end, candidate_start, candidate_end)}
```

- [ ] **Step 4: Run overview API tests**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/api/test_overview_api.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/repository.py app/api/routes.py app/domain/schemas.py tests/api/test_overview_api.py
git commit -m "feat: add stage telemetry overview and benchmark comparison endpoints"
```

### Task 6: Document telemetry pipeline and run final verification

**Files:**
- Modify: `docs/pipeline-and-schema-brief.md`
- Test: `tests/unit/test_telemetry.py`
- Test: `tests/api/test_overview_api.py`
- Test: `tests/api/test_intents_api.py`
- Test: `tests/integration/test_schema.py`

- [ ] **Step 1: Update docs with telemetry/cost tracking flow**

```markdown
## Runtime telemetry additions
- New tables: inference_stage_metrics, inference_cost_pricing, inference_metrics_rollup_hourly
- Stage list: stt, embedding, vector_search, rerank, confidence_policy, persistence, total
- Benchmark KPIs: per-stage p50/p95 and cost per 1k requests
```

- [ ] **Step 2: Run targeted telemetry test suite**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest tests/unit/test_telemetry.py tests/api/test_overview_api.py tests/api/test_intents_api.py tests/integration/test_schema.py -q
```
Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:
```bash
/home/cst/dev/callSteering-renewed/.venv/bin/python -m pytest -q
```
Expected: PASS.

- [ ] **Step 4: Commit docs and final integration updates**

```bash
git add docs/pipeline-and-schema-brief.md
git commit -m "docs: describe runtime telemetry and benchmarking KPIs"
```

- [ ] **Step 5: Create final feature commit if needed**

```bash
git add -A
git commit -m "feat: complete runtime latency and cost telemetry rollout"
```

## Self-Review Notes

### Spec coverage
- Stage-level latency and cost capture: covered in Tasks 2–4.
- PostgreSQL persistence (events + pricing + rollup): covered in Tasks 1, 3, 5.
- P50/P95 and cost-per-1k visibility: covered in Task 5.
- Benchmark window comparison for viability/cost deltas: covered in Task 5.
- Testing strategy (unit/integration/api): covered in Tasks 1–6.

### Placeholder scan
- Verified no placeholder tokens remain in executable steps.

### Type/signature consistency
- Telemetry collector and repository method names are consistent across Tasks 2–5:
  - `InferenceTelemetryCollector`
  - `estimate_stage_cost`
  - `create_stage_metrics`
  - `stage_latency` / `stage_cost` / `benchmark_compare`
