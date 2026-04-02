from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.db.models import InferenceRequest, InferenceResult
from app.services.repository import InferenceRepository


class _FakeDb:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, item: object) -> None:
        self.added.append(item)

    def flush(self) -> None:
        return None

    def execute(self, statement):
        _ = statement
        return SimpleNamespace(all=lambda: [])


def test_summary_delta_computation(monkeypatch) -> None:
    repo = InferenceRepository(_FakeDb())
    values = iter([(12, 0.8), (6, 0.4)])
    monkeypatch.setattr(repo, '_window_metrics', lambda start, end: next(values))

    result = repo.summary(
        datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert result['total_inferences'] == 12
    assert result['avg_confidence'] == 0.8
    assert result['total_inferences_delta_pct'] == 100.0
    assert result['avg_confidence_delta_pct'] == 100.0


def test_summary_delta_none_when_previous_zero(monkeypatch) -> None:
    repo = InferenceRepository(_FakeDb())
    values = iter([(4, 0.5), (0, 0.0)])
    monkeypatch.setattr(repo, '_window_metrics', lambda start, end: next(values))

    result = repo.summary(
        datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert result['total_inferences_delta_pct'] is None
    assert result['avg_confidence_delta_pct'] is None


def test_intent_distribution_unmatched_and_percentage(monkeypatch) -> None:
    captured_statement: dict[str, object] = {}

    class _RowsDb:
        def execute(self, statement):
            captured_statement['query'] = statement
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(intent_code='billing', count=7),
                    SimpleNamespace(intent_code='unmatched', count=3),
                ]
            )

    repo = InferenceRepository(_RowsDb())
    monkeypatch.setattr(repo, '_window_metrics', lambda start, end: (10, 0.0))
    items = repo.intent_distribution(
        datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert items == [
        {'intent_code': 'billing', 'count': 7, 'percentage': 70.0},
        {'intent_code': 'unmatched', 'count': 3, 'percentage': 30.0},
    ]
    order_by_clauses = list(captured_statement['query']._order_by_clauses)
    assert len(order_by_clauses) == 2
    assert 'count(inference_requests.id) DESC' in str(order_by_clauses[0])
    assert 'coalesce(intents.intent_code' in str(order_by_clauses[1])
    assert str(order_by_clauses[1]).endswith('ASC')


def test_recent_activity_ordering_and_limit() -> None:
    captured_statement: dict[str, object] = {}

    class _RowsDb:
        def execute(self, statement):
            captured_statement['query'] = statement
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(
                        timestamp=datetime(2026, 4, 1, 11, 59, tzinfo=timezone.utc),
                        input_snippet='newest',
                        predicted_intent='billing',
                        confidence=0.9,
                    ),
                ]
            )

    repo = InferenceRepository(_RowsDb())
    items = repo.recent_activity(
        datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
        1,
    )
    query_sql = str(captured_statement['query'])
    assert 'LIMIT :param_1' in query_sql
    assert 'ORDER BY inference_requests.created_at DESC' in query_sql
    assert len(items) == 1
    assert items[0]['input_snippet'] == 'newest'


def test_create_request_and_result_persist_matched_records() -> None:
    db = _FakeDb()
    repo = InferenceRepository(db)
    intent_id = uuid4()
    request = repo.create_request(
        external_request_id='req-1',
        language_code='tr',
        audio_uri='file:///tmp/audio.wav',
        transcript='balance question',
        status='matched',
        processing_ms=120,
    )
    repo.create_result(
        request_id=request.id,
        predicted_intent_id=intent_id,
        confidence=0.89,
        top_k_json=[{'intent_code': 'balance_inquiry', 'score': 0.89}],
    )

    assert isinstance(db.added[0], InferenceRequest)
    assert db.added[0].status == 'matched'
    assert db.added[0].external_request_id == 'req-1'
    assert isinstance(db.added[1], InferenceResult)
    assert db.added[1].predicted_intent_id == intent_id
    assert db.added[1].policy_version == 'v1'


def test_create_request_and_result_persist_low_confidence_records() -> None:
    db = _FakeDb()
    repo = InferenceRepository(db)
    request = repo.create_request(
        external_request_id=None,
        language_code='tr',
        audio_uri='file:///tmp/audio.wav',
        transcript='unclear request',
        status='low_confidence',
        processing_ms=120,
    )
    repo.create_result(
        request_id=request.id,
        predicted_intent_id=None,
        confidence=0.42,
        top_k_json=[{'intent_code': 'balance_inquiry', 'score': 0.42}],
    )

    assert isinstance(db.added[0], InferenceRequest)
    assert db.added[0].status == 'low_confidence'
    assert isinstance(db.added[1], InferenceResult)
    assert db.added[1].predicted_intent_id is None
    assert db.added[1].confidence == 0.42
