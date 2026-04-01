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

    def by_intent_id(self, intent_id: UUID | str) -> list[IntentEmbedding]:
        return list(self.db.scalars(select(IntentEmbedding).where(IntentEmbedding.intent_id == intent_id)).all())

    def upsert_for_intent(self, intent_id: UUID | str, model_name: str, embedding: list[float]) -> IntentEmbedding:
        record = self.db.scalar(
            select(IntentEmbedding).where(
                IntentEmbedding.intent_id == intent_id,
                IntentEmbedding.model_name == model_name,
            )
        )
        now = _utc_now()
        norm = sum(item * item for item in embedding) ** 0.5
        if record is None:
            record = IntentEmbedding(
                intent_id=intent_id,
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

    def by_intent_id(self, intent_id: str) -> list[IntentUtterance]:
        return list(self.db.scalars(select(IntentUtterance).where(IntentUtterance.intent_id == intent_id)).all())
