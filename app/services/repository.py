from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, Select
from sqlalchemy.orm import Session

from app.db.models import InferenceRequest, InferenceResult, Intent, IntentEmbedding, IntentUtterance


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
