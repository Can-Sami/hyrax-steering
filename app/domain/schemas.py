from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class IntentCandidate:
    intent_id: UUID
    intent_code: str
    score: float


@dataclass(frozen=True)
class RerankedIntentCandidate:
    intent_id: UUID
    intent_code: str
    semantic_score: float
    reranker_score: float


@dataclass(frozen=True)
class ConfidenceResult:
    top_candidate: IntentCandidate | None
    top_k: tuple[IntentCandidate, ...]
    confidence: float
    is_low_confidence: bool


@dataclass(frozen=True)
class StageLatencyStats:
    stage_name: str
    request_count: int
    error_count: int
    p50_ms: float
    p95_ms: float
    avg_ms: float


@dataclass(frozen=True)
class StageCostStats:
    stage_name: str
    request_count: int
    total_estimated_cost_usd: float
    cost_per_1k_requests: float


@dataclass(frozen=True)
class BenchmarkStageComparison:
    stage_name: str
    baseline_p95_ms: float
    candidate_p95_ms: float
    p95_delta_pct: float | None
    baseline_cost_per_1k_requests: float
    candidate_cost_per_1k_requests: float
    cost_per_1k_delta_pct: float | None
