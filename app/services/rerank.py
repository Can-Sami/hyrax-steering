from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.domain.schemas import RerankedIntentCandidate
from app.services.repository import IntentRepository
from app.services.similarity import SimilaritySearchService
from app.workers.pipeline import RerankerProvider


class TwoStageIntentSearchService:
    def __init__(self, db: Session, reranker_provider: RerankerProvider) -> None:
        self.db = db
        self.reranker_provider = reranker_provider
        self.similarity_service = SimilaritySearchService(db)
        self.intent_repository = IntentRepository(db)

    def top_k(
        self,
        query: str,
        embedding: Sequence[float],
        k: int = 5,
        language_code: str = 'tr',
    ) -> list[RerankedIntentCandidate]:
        semantic_candidates = self.similarity_service.top_k(embedding=embedding, k=k, language_code=language_code)
        if not semantic_candidates:
            return []

        intent_ids = [candidate.intent_id for candidate in semantic_candidates]
        descriptions = self.intent_repository.get_descriptions_by_ids(intent_ids)
        documents = [descriptions.get(candidate.intent_id, '') for candidate in semantic_candidates]
        rerank_scores = self.reranker_provider.score_pairs(query=query, documents=documents)

        items = [
            RerankedIntentCandidate(
                intent_id=candidate.intent_id,
                intent_code=candidate.intent_code,
                semantic_score=candidate.score,
                reranker_score=rerank_scores[index],
            )
            for index, candidate in enumerate(semantic_candidates)
        ]
        return sorted(items, key=lambda item: item.reranker_score, reverse=True)
