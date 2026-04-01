from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.models import Intent, IntentEmbedding
from app.domain.schemas import IntentCandidate


class SimilaritySearchService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def top_k(self, embedding: Sequence[float], k: int = 5, language_code: str = 'tr') -> list[IntentCandidate]:
        if k < 1:
            raise ValueError('k must be >= 1')

        _ = language_code
        query: Select = (
            select(
                Intent.id,
                Intent.intent_code,
                (1 - IntentEmbedding.embedding.cosine_distance(embedding)).label('score'),
            )
            .join(IntentEmbedding, IntentEmbedding.intent_id == Intent.id)
            .where(Intent.is_active.is_(True))
            .order_by(IntentEmbedding.embedding.cosine_distance(embedding))
            .limit(k)
        )

        rows = self.db.execute(query).all()
        return [IntentCandidate(intent_id=row.id, intent_code=row.intent_code, score=float(row.score)) for row in rows]
