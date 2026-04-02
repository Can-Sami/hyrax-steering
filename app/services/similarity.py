from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.db.models import Intent, IntentEmbedding, IntentUtterance
from app.domain.schemas import IntentCandidate


class SimilaritySearchService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def top_k(self, embedding: Sequence[float], k: int = 5, language_code: str = 'tr') -> list[IntentCandidate]:
        if k < 1:
            raise ValueError('k must be >= 1')

        _ = language_code
        score_expr = (1 - IntentEmbedding.embedding.cosine_distance(embedding)).label('score')
        query: Select = (
            select(
                Intent.id,
                Intent.intent_code,
                func.max(score_expr).label('score'),
            )
            .join(IntentUtterance, IntentUtterance.intent_id == Intent.id)
            .join(IntentEmbedding, IntentEmbedding.utterance_id == IntentUtterance.id)
            .where(
                Intent.is_active.is_(True),
                IntentUtterance.language_code == language_code,
            )
            .group_by(Intent.id, Intent.intent_code)
            .order_by(func.max(score_expr).desc())
            .limit(k)
        )

        rows = self.db.execute(query).all()
        return [IntentCandidate(intent_id=row.id, intent_code=row.intent_code, score=float(row.score)) for row in rows]
