from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class IntentCandidate:
    intent_id: UUID
    intent_code: str
    score: float


@dataclass(frozen=True)
class ConfidenceResult:
    top_candidate: IntentCandidate | None
    top_k: tuple[IntentCandidate, ...]
    confidence: float
    is_low_confidence: bool
