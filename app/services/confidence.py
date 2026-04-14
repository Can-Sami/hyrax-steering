from app.domain.schemas import ConfidenceResult, IntentCandidate


class ConfidencePolicy:
    def __init__(self, threshold: float = 0.65, min_margin: float = 0.01) -> None:
        self.threshold = threshold
        self.min_margin = min_margin

    def evaluate(self, candidates: list[IntentCandidate]) -> ConfidenceResult:
        if not candidates:
            return ConfidenceResult(
                top_candidate=None,
                top_k=(),
                confidence=0.0,
                is_low_confidence=True,
            )

        top = candidates[0]
        runner_up_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top.score - runner_up_score

        is_low_confidence = top.score < self.threshold or margin < self.min_margin

        return ConfidenceResult(
            top_candidate=top,
            top_k=tuple(candidates),
            confidence=top.score,
            is_low_confidence=is_low_confidence,
        )
