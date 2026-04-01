from fastapi.testclient import TestClient

from app.main import app
from app.api import routes
from app.domain.schemas import IntentCandidate


client = TestClient(app)


def test_inference_contract(monkeypatch) -> None:
    class _FakeSimilarity:
        def __init__(self, db) -> None:
            self.db = db

        def top_k(self, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (embedding, k, language_code)
            return [IntentCandidate(intent_id=routes.uuid4(), intent_code='fatura_sorgulama', score=0.84)]

    class _FakeDb:
        pass

    monkeypatch.setattr(routes, 'SimilaritySearchService', _FakeSimilarity)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    files = {'audio_file': ('sample.wav', b'fake-audio', 'audio/wav')}
    data = {'language_hint': 'tr'}

    try:
        response = client.post('/api/v1/inference/intent', files=files, data=data)
        assert response.status_code == 200

        body = response.json()
        assert {'request_id', 'intent_code', 'confidence', 'match_status', 'transcript', 'top_candidates', 'processing_ms'} <= set(body.keys())
    finally:
        app.dependency_overrides.clear()
