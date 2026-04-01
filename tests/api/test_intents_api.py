from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.api import routes
from app.domain.schemas import IntentCandidate
from app.workers.pipeline import EmbeddingResult


class _FakeDb:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def flush(self) -> None:
        return None

    def delete(self, _: object) -> None:
        return None


class _FakeIntent:
    def __init__(self, intent_code: str, description: str) -> None:
        self.id = uuid4()
        self.intent_code = intent_code
        self.description = description
        self.is_active = True
        self.created_at = routes.datetime.now(routes.timezone.utc)
        self.updated_at = routes.datetime.now(routes.timezone.utc)


def _fake_embed(text: str) -> list[float]:
    if 'fatura' in text:
        return [0.99] + [0.0] * 1023
    if 'tarife' in text:
        return [0.1, 0.95] + [0.0] * 1022
    return [0.2] * 1024


def test_intent_crud_and_search(monkeypatch) -> None:
    store: dict[str, _FakeIntent] = {}

    class FakeIntentRepository:
        def __init__(self, db) -> None:
            self.db = db

        def list_all(self):
            return list(store.values())

        def list_active(self):
            return [item for item in store.values() if item.is_active]

        def get_by_id(self, intent_id):
            return store.get(str(intent_id))

        def create(self, intent_code: str, description: str):
            intent = _FakeIntent(intent_code=intent_code, description=description)
            store[str(intent.id)] = intent
            return intent

        def update(self, intent, intent_code: str, description: str):
            intent.intent_code = intent_code
            intent.description = description
            return intent

        def delete(self, intent):
            store.pop(str(intent.id), None)

    class FakeEmbeddingRepository:
        def __init__(self, db) -> None:
            self.db = db

        def upsert_for_intent(self, intent_id, model_name: str, embedding: list[float]):
            _ = (intent_id, model_name, embedding)
            return None

    class FakeSimilaritySearchService:
        def __init__(self, db) -> None:
            self.db = db

        def top_k(self, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (embedding, language_code)
            intents = list(store.values())[:k]
            return [
                IntentCandidate(intent_id=item.id, intent_code=item.intent_code, score=0.9 - idx * 0.1)
                for idx, item in enumerate(intents)
            ]

    monkeypatch.setattr(routes.pipeline.embedding_provider, 'embed', lambda text: EmbeddingResult(vector=_fake_embed(text)))
    monkeypatch.setattr(routes, 'IntentRepository', FakeIntentRepository)
    monkeypatch.setattr(routes, 'EmbeddingRepository', FakeEmbeddingRepository)
    monkeypatch.setattr(routes, 'SimilaritySearchService', FakeSimilaritySearchService)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    client = TestClient(app)

    create_response = client.post(
        '/api/v1/intents',
        json={'intent_code': f'fatura_{uuid4().hex[:8]}', 'description': 'Fatura ile ilgili islemler'},
    )
    assert create_response.status_code == 200
    created = create_response.json()
    intent_id = created['id']

    list_response = client.get('/api/v1/intents')
    assert list_response.status_code == 200
    assert any(item['id'] == intent_id for item in list_response.json()['items'])

    update_response = client.put(
        f'/api/v1/intents/{intent_id}',
        json={'intent_code': f'tarife_{uuid4().hex[:8]}', 'description': 'Tarife degisikligi islemleri'},
    )
    assert update_response.status_code == 200

    search_response = client.post('/api/v1/intents/search', json={'query': 'tarife degistirmek istiyorum', 'k': 3})
    assert search_response.status_code == 200
    assert 'items' in search_response.json()
    assert len(search_response.json()['items']) >= 1

    delete_response = client.delete(f'/api/v1/intents/{intent_id}')
    assert delete_response.status_code == 200
    assert delete_response.json()['status'] == 'deleted'
    app.dependency_overrides.clear()
