from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import case, func, select, Select
from sqlalchemy.orm import Session

from app.db.models import (
    InferenceCostPricing,
    InferenceRequest,
    InferenceResult,
    InferenceStageMetric,
    Intent,
    IntentEmbedding,
    IntentUtterance,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_active(self) -> list[Intent]:
        return list(self.db.scalars(select(Intent).where(Intent.is_active.is_(True))).all())

    def list_all(self) -> list[Intent]:
        return list(self.db.scalars(select(Intent).order_by(Intent.created_at.desc())).all())

    def get_by_id(self, intent_id: UUID | str) -> Intent | None:
        return self.db.get(Intent, intent_id)

    def create(self, intent_code: str, description: str) -> Intent:
        now = _utc_now()
        intent = Intent(
            intent_code=intent_code,
            description=description,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self.db.add(intent)
        self.db.flush()
        return intent

    def update(self, intent: Intent, intent_code: str, description: str) -> Intent:
        intent.intent_code = intent_code
        intent.description = description
        intent.updated_at = _utc_now()
        self.db.flush()
        return intent

    def delete(self, intent: Intent) -> None:
        self.db.delete(intent)

    def get_descriptions_by_ids(self, intent_ids: list[UUID | str]) -> dict[UUID, str]:
        if not intent_ids:
            return {}
        query: Select = select(Intent.id, Intent.description).where(Intent.id.in_(intent_ids))
        rows = self.db.execute(query).all()
        return {row.id: row.description for row in rows}


class EmbeddingRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def by_utterance_id(self, utterance_id: UUID | str) -> list[IntentEmbedding]:
        return list(self.db.scalars(select(IntentEmbedding).where(IntentEmbedding.utterance_id == utterance_id)).all())

    def upsert_for_utterance(self, utterance_id: UUID | str, model_name: str, embedding: list[float]) -> IntentEmbedding:
        record = self.db.scalar(
            select(IntentEmbedding).where(
                IntentEmbedding.utterance_id == utterance_id,
                IntentEmbedding.model_name == model_name,
            )
        )
        now = _utc_now()
        norm = sum(item * item for item in embedding) ** 0.5
        if record is None:
            record = IntentEmbedding(
                utterance_id=utterance_id,
                model_name=model_name,
                embedding=embedding,
                norm=norm,
                created_at=now,
                updated_at=now,
            )
            self.db.add(record)
        else:
            record.embedding = embedding
            record.norm = norm
            record.updated_at = now
        self.db.flush()
        return record


class UtteranceRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def by_intent_id(self, intent_id: UUID | str) -> list[IntentUtterance]:
        return list(self.db.scalars(select(IntentUtterance).where(IntentUtterance.intent_id == intent_id)).all())

    def create(self, intent_id: UUID | str, language_code: str, text: str, source: str) -> IntentUtterance:
        now = _utc_now()
        utterance = IntentUtterance(
            intent_id=intent_id,
            language_code=language_code,
            text=text,
            source=source,
            created_at=now,
            updated_at=now,
        )
        self.db.add(utterance)
        self.db.flush()
        return utterance

    def get_by_id(self, utterance_id: UUID | str) -> IntentUtterance | None:
        return self.db.get(IntentUtterance, utterance_id)

    def update(self, utterance: IntentUtterance, language_code: str, text: str, source: str) -> IntentUtterance:
        utterance.language_code = language_code
        utterance.text = text
        utterance.source = source
        utterance.updated_at = _utc_now()
        self.db.flush()
        return utterance

    def delete(self, utterance: IntentUtterance) -> None:
        self.db.delete(utterance)


class InferenceRepository:
    POLICY_VERSION = 'v1'

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_intent_by_code(self, intent_code: str) -> Intent | None:
        return self.db.scalar(select(Intent).where(Intent.intent_code == intent_code))

    def create_request(
        self,
        *,
        external_request_id: str | None,
        language_code: str,
        audio_uri: str,
        transcript: str | None,
        status: str,
        processing_ms: int | None,
    ) -> InferenceRequest:
        now = _utc_now()
        inference_request = InferenceRequest(
            external_request_id=external_request_id,
            language_code=language_code,
            audio_uri=audio_uri,
            transcript=transcript,
            status=status,
            processing_ms=processing_ms,
            created_at=now,
            updated_at=now,
        )
        self.db.add(inference_request)
        self.db.flush()
        return inference_request

    def create_result(
        self,
        *,
        request_id,
        predicted_intent_id,
        confidence: float,
        top_k_json: list[dict[str, str | float]],
    ) -> InferenceResult:
        result = InferenceResult(
            request_id=request_id,
            predicted_intent_id=predicted_intent_id,
            confidence=confidence,
            top_k_json=top_k_json,
            policy_version=self.POLICY_VERSION,
            created_at=_utc_now(),
        )
        self.db.add(result)
        self.db.flush()
        return result

    def _window_metrics(self, start_at: datetime, end_at: datetime) -> tuple[int, float]:
        total_inferences = self.db.scalar(
            select(func.count(InferenceRequest.id)).where(
                InferenceRequest.created_at >= start_at,
                InferenceRequest.created_at < end_at,
            )
        )
        avg_confidence = self.db.scalar(
            select(func.avg(InferenceResult.confidence))
            .join(InferenceRequest, InferenceResult.request_id == InferenceRequest.id)
            .where(
                InferenceRequest.created_at >= start_at,
                InferenceRequest.created_at < end_at,
            )
        )
        return int(total_inferences or 0), float(avg_confidence or 0.0)

    @staticmethod
    def _delta_pct(current: float, previous: float) -> float | None:
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100.0

    def summary(self, start_at: datetime, end_at: datetime) -> dict[str, int | float | None]:
        total_inferences, avg_confidence = self._window_metrics(start_at, end_at)
        duration = end_at - start_at
        previous_start = start_at - duration
        previous_total, previous_avg = self._window_metrics(previous_start, start_at)

        return {
            'total_inferences': total_inferences,
            'avg_confidence': avg_confidence,
            'total_inferences_delta_pct': self._delta_pct(float(total_inferences), float(previous_total)),
            'avg_confidence_delta_pct': self._delta_pct(avg_confidence, previous_avg),
        }

    def intent_distribution(self, start_at: datetime, end_at: datetime) -> list[dict[str, int | float | str]]:
        total_inferences, _ = self._window_metrics(start_at, end_at)
        label = func.coalesce(Intent.intent_code, 'unmatched')
        rows = self.db.execute(
            select(
                label.label('intent_code'),
                func.count(InferenceRequest.id).label('count'),
            )
            .select_from(InferenceRequest)
            .outerjoin(InferenceResult, InferenceResult.request_id == InferenceRequest.id)
            .outerjoin(Intent, InferenceResult.predicted_intent_id == Intent.id)
            .where(
                InferenceRequest.created_at >= start_at,
                InferenceRequest.created_at < end_at,
            )
            .group_by(label)
            .order_by(func.count(InferenceRequest.id).desc(), label.asc())
        ).all()

        items: list[dict[str, int | float | str]] = []
        for row in rows:
            count = int(row.count or 0)
            percentage = (count / total_inferences * 100.0) if total_inferences else 0.0
            items.append(
                {
                    'intent_code': row.intent_code,
                    'count': count,
                    'percentage': percentage,
                }
            )
        return items

    def recent_activity(self, start_at: datetime, end_at: datetime, limit: int) -> list[dict[str, object]]:
        rows = self.db.execute(
            select(
                InferenceRequest.created_at.label('timestamp'),
                InferenceRequest.transcript.label('input_snippet'),
                Intent.intent_code.label('predicted_intent'),
                InferenceResult.confidence.label('confidence'),
            )
            .select_from(InferenceRequest)
            .outerjoin(InferenceResult, InferenceResult.request_id == InferenceRequest.id)
            .outerjoin(Intent, InferenceResult.predicted_intent_id == Intent.id)
            .where(
                InferenceRequest.created_at >= start_at,
                InferenceRequest.created_at < end_at,
            )
            .order_by(InferenceRequest.created_at.desc())
            .limit(limit)
        ).all()

        return [
            {
                'timestamp': row.timestamp.isoformat(),
                'input_snippet': row.input_snippet,
                'predicted_intent': row.predicted_intent,
                'confidence': row.confidence,
            }
            for row in rows
        ]

    def stage_latency(self, start_at: datetime, end_at: datetime) -> list[dict[str, object]]:
        rows = self.db.execute(
            select(
                InferenceStageMetric.stage_name,
                func.count(InferenceStageMetric.id).label('request_count'),
                func.sum(case((InferenceStageMetric.status == 'error', 1), else_=0)).label('error_count'),
                func.percentile_cont(0.50).within_group(InferenceStageMetric.duration_ms).label('p50_ms'),
                func.percentile_cont(0.95).within_group(InferenceStageMetric.duration_ms).label('p95_ms'),
                func.avg(InferenceStageMetric.duration_ms).label('avg_ms'),
            ).where(
                InferenceStageMetric.created_at >= start_at,
                InferenceStageMetric.created_at < end_at,
            ).group_by(
                InferenceStageMetric.stage_name,
            ).order_by(
                InferenceStageMetric.stage_name.asc(),
            )
        ).all()

        return [
            {
                'stage_name': row.stage_name,
                'request_count': int(row.request_count or 0),
                'error_count': int(row.error_count or 0),
                'p50_ms': float(row.p50_ms or 0.0),
                'p95_ms': float(row.p95_ms or 0.0),
                'avg_ms': float(row.avg_ms or 0.0),
            }
            for row in rows
        ]

    def stage_cost(self, start_at: datetime, end_at: datetime) -> list[dict[str, object]]:
        rows = self.db.execute(
            select(
                InferenceStageMetric.stage_name,
                func.count(InferenceStageMetric.id).label('request_count'),
                func.sum(func.coalesce(InferenceStageMetric.estimated_cost_usd, 0.0)).label('total_estimated_cost_usd'),
            ).where(
                InferenceStageMetric.created_at >= start_at,
                InferenceStageMetric.created_at < end_at,
            ).group_by(
                InferenceStageMetric.stage_name,
            ).order_by(
                InferenceStageMetric.stage_name.asc(),
            )
        ).all()

        items: list[dict[str, object]] = []
        for row in rows:
            request_count = int(row.request_count or 0)
            total_cost = float(row.total_estimated_cost_usd or 0.0)
            cost_per_1k = (total_cost / request_count * 1000.0) if request_count else 0.0
            items.append(
                {
                    'stage_name': row.stage_name,
                    'request_count': request_count,
                    'total_estimated_cost_usd': total_cost,
                    'cost_per_1k_requests': cost_per_1k,
                }
            )
        return items

    def benchmark_compare(
        self,
        baseline_start: datetime,
        baseline_end: datetime,
        candidate_start: datetime,
        candidate_end: datetime,
    ) -> list[dict[str, object]]:
        baseline_latency = {item['stage_name']: item for item in self.stage_latency(baseline_start, baseline_end)}
        baseline_cost = {item['stage_name']: item for item in self.stage_cost(baseline_start, baseline_end)}
        candidate_latency = {item['stage_name']: item for item in self.stage_latency(candidate_start, candidate_end)}
        candidate_cost = {item['stage_name']: item for item in self.stage_cost(candidate_start, candidate_end)}

        all_stage_names = sorted(
            set(baseline_latency.keys())
            | set(baseline_cost.keys())
            | set(candidate_latency.keys())
            | set(candidate_cost.keys())
        )
        items: list[dict[str, object]] = []
        for stage_name in all_stage_names:
            baseline_p95 = float(baseline_latency.get(stage_name, {}).get('p95_ms', 0.0))
            candidate_p95 = float(candidate_latency.get(stage_name, {}).get('p95_ms', 0.0))
            baseline_cost_per_1k = float(baseline_cost.get(stage_name, {}).get('cost_per_1k_requests', 0.0))
            candidate_cost_per_1k = float(candidate_cost.get(stage_name, {}).get('cost_per_1k_requests', 0.0))
            items.append(
                {
                    'stage_name': stage_name,
                    'baseline_p95_ms': baseline_p95,
                    'candidate_p95_ms': candidate_p95,
                    'p95_delta_pct': self._delta_pct(candidate_p95, baseline_p95),
                    'baseline_cost_per_1k_requests': baseline_cost_per_1k,
                    'candidate_cost_per_1k_requests': candidate_cost_per_1k,
                    'cost_per_1k_delta_pct': self._delta_pct(candidate_cost_per_1k, baseline_cost_per_1k),
                }
            )
        return items


class InferenceTelemetryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_stage_metrics(self, request_id: UUID | str, stage_rows: list[dict[str, Any]]) -> None:
        if not hasattr(self.db, 'add') or not hasattr(self.db, 'flush'):
            return
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

    def get_active_price(
        self,
        *,
        provider: str,
        model_name: str,
        unit_type: str,
        at: datetime,
    ) -> InferenceCostPricing | None:
        if not hasattr(self.db, 'scalar'):
            return None
        return self.db.scalar(
            select(InferenceCostPricing)
            .where(
                InferenceCostPricing.provider == provider,
                InferenceCostPricing.model_name == model_name,
                InferenceCostPricing.unit_type == unit_type,
                InferenceCostPricing.effective_from <= at,
                func.coalesce(InferenceCostPricing.effective_to, at) >= at,
            )
            .order_by(InferenceCostPricing.effective_from.desc())
            .limit(1)
        )

    def list_active_prices(
        self,
        *,
        provider: str,
        model_name: str,
        at: datetime,
    ) -> list[InferenceCostPricing]:
        if not hasattr(self.db, 'scalars'):
            return []
        return list(
            self.db.scalars(
                select(InferenceCostPricing).where(
                    InferenceCostPricing.provider == provider,
                    InferenceCostPricing.model_name == model_name,
                    InferenceCostPricing.effective_from <= at,
                    func.coalesce(InferenceCostPricing.effective_to, at) >= at,
                )
            ).all()
        )
