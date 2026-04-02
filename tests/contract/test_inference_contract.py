from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.api import routes
from app.domain.schemas import ConfidenceResult, IntentCandidate


client = TestClient(app)


def test_inference_contract_persists_matched(monkeypatch) -> None:
    class _FakeSimilarity:
        def __init__(self, db) -> None:
            self.db = db

        def top_k(self, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (embedding, k, language_code)
            return [IntentCandidate(intent_id=routes.uuid4(), intent_code='fatura_sorgulama', score=0.84)]

    class _FakeDb:
        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    captured: dict[str, object] = {}

    class _FakeInferenceRequest:
        def __init__(self) -> None:
            self.id = uuid4()

    class _FakeIntent:
        def __init__(self, intent_id) -> None:
            self.id = intent_id

    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def get_intent_by_code(self, intent_code: str):
            captured['intent_lookup'] = intent_code
            return _FakeIntent(uuid4())

        def create_request(self, **kwargs):
            captured['request'] = kwargs
            return _FakeInferenceRequest()

        def create_result(self, **kwargs):
            captured['result'] = kwargs
            return None

    monkeypatch.setattr(routes, 'SimilaritySearchService', _FakeSimilarity)
    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    monkeypatch.setattr(routes.storage, 'store', lambda filename, content: f'/tmp/{filename}')
    monkeypatch.setattr(
        routes.confidence_policy,
        'evaluate',
        lambda candidates: ConfidenceResult(
            top_candidate=candidates[0],
            top_k=tuple(candidates),
            confidence=0.84,
            is_low_confidence=False,
        ),
    )
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    files = {'audio_file': ('sample.wav', b'fake-audio', 'audio/wav')}
    data = {'language_hint': 'tr', 'request_id': 'req-123'}

    try:
        response = client.post('/api/v1/inference/intent', files=files, data=data)
        assert response.status_code == 200

        body = response.json()
        assert {'request_id', 'intent_code', 'confidence', 'match_status', 'transcript', 'top_candidates', 'processing_ms'} <= set(body.keys())
        assert body['match_status'] == 'matched'
        assert captured['intent_lookup'] == 'fatura_sorgulama'
        assert captured['request']['external_request_id'] == 'req-123'
        assert captured['request']['language_code'] == 'tr'
        assert captured['request']['status'] == 'matched'
        assert captured['result']['confidence'] == 0.84
        assert captured['result']['predicted_intent_id'] is not None
        assert isinstance(captured['result']['top_k_json'], list)
    finally:
        app.dependency_overrides.clear()


def test_inference_contract_persists_low_confidence(monkeypatch) -> None:
    class _FakeSimilarity:
        def __init__(self, db) -> None:
            self.db = db

        def top_k(self, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (embedding, k, language_code)
            return [IntentCandidate(intent_id=routes.uuid4(), intent_code='fatura_sorgulama', score=0.5)]

    class _FakeDb:
        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    captured: dict[str, object] = {}

    class _FakeInferenceRequest:
        def __init__(self) -> None:
            self.id = uuid4()

    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def get_intent_by_code(self, intent_code: str):
            captured['intent_lookup'] = intent_code
            return None

        def create_request(self, **kwargs):
            captured['request'] = kwargs
            return _FakeInferenceRequest()

        def create_result(self, **kwargs):
            captured['result'] = kwargs
            return None

    monkeypatch.setattr(routes, 'SimilaritySearchService', _FakeSimilarity)
    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    monkeypatch.setattr(routes.storage, 'store', lambda filename, content: f'/tmp/{filename}')
    monkeypatch.setattr(
        routes.confidence_policy,
        'evaluate',
        lambda candidates: ConfidenceResult(
            top_candidate=candidates[0],
            top_k=tuple(candidates),
            confidence=0.5,
            is_low_confidence=True,
        ),
    )
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    files = {'audio_file': ('sample.wav', b'fake-audio', 'audio/wav')}
    data = {'language_hint': 'tr'}

    try:
        response = client.post('/api/v1/inference/intent', files=files, data=data)
        assert response.status_code == 200
        body = response.json()
        assert body['match_status'] == 'low_confidence'
        assert body['intent_code'] is None
        assert captured['request']['status'] == 'low_confidence'
        assert captured['request']['external_request_id'] == body['request_id']
        assert captured['result']['predicted_intent_id'] is None
        assert captured['result']['confidence'] == 0.5
    finally:
        app.dependency_overrides.clear()
