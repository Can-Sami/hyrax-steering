from uuid import uuid4

import pytest

from app.domain.schemas import IntentCandidate
from app.services.confidence import ConfidencePolicy


@pytest.mark.parametrize(
    ('scores', 'threshold', 'min_margin', 'expected_low_confidence'),
    [
        ([0.92, 0.71], 0.65, 0.05, False),
        ([0.60, 0.51], 0.65, 0.05, True),
        ([0.70, 0.68], 0.65, 0.05, True),
        ([], 0.65, 0.05, True),
    ],
)
def test_confidence_policy(scores: list[float], threshold: float, min_margin: float, expected_low_confidence: bool) -> None:
    policy = ConfidencePolicy(threshold=threshold, min_margin=min_margin)
    candidates = [
        IntentCandidate(intent_id=uuid4(), intent_code=f'intent_{idx}', score=score)
        for idx, score in enumerate(scores)
    ]

    result = policy.evaluate(candidates)
    assert result.is_low_confidence is expected_low_confidence
    if scores:
        assert result.confidence == scores[0]
    else:
        assert result.confidence == 0.0
