from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Intent, IntentEmbedding, IntentUtterance


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
