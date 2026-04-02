from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.api import routes
from app.api.errors import AppError
from app.domain.schemas import IntentCandidate, RerankedIntentCandidate
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


class _FakeUtterance:
    def __init__(self, intent_id, language_code: str, text: str, source: str) -> None:
        self.id = uuid4()
        self.intent_id = intent_id
        self.language_code = language_code
        self.text = text
        self.source = source
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
    utterance_store: dict[str, _FakeUtterance] = {}

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

        def upsert_for_utterance(self, utterance_id, model_name: str, embedding: list[float]):
            _ = (utterance_id, model_name, embedding)
            return None

    class FakeUtteranceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def by_intent_id(self, intent_id):
            return [item for item in utterance_store.values() if str(item.intent_id) == str(intent_id)]

        def create(self, intent_id, language_code: str, text: str, source: str):
            utterance = _FakeUtterance(intent_id=intent_id, language_code=language_code, text=text, source=source)
            utterance_store[str(utterance.id)] = utterance
            return utterance

        def get_by_id(self, utterance_id):
            return utterance_store.get(str(utterance_id))

        def update(self, utterance, language_code: str, text: str, source: str):
            utterance.language_code = language_code
            utterance.text = text
            utterance.source = source
            utterance.updated_at = routes.datetime.now(routes.timezone.utc)
            return utterance

        def delete(self, utterance):
            utterance_store.pop(str(utterance.id), None)

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

    class FakeTwoStageIntentSearchService:
        def __init__(self, db, reranker_provider) -> None:
            _ = (db, reranker_provider)

        def top_k(self, query: str, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (query, embedding, k, language_code)
            intents = list(store.values())[:k]
            return [
                RerankedIntentCandidate(
                    intent_id=item.id,
                    intent_code=item.intent_code,
                    semantic_score=0.7 - idx * 0.1,
                    reranker_score=0.9 - idx * 0.1,
                )
                for idx, item in enumerate(intents)
            ]

    monkeypatch.setattr(routes.pipeline.embedding_provider, 'embed', lambda text: EmbeddingResult(vector=_fake_embed(text)))
    monkeypatch.setattr(routes, 'IntentRepository', FakeIntentRepository)
    monkeypatch.setattr(routes, 'EmbeddingRepository', FakeEmbeddingRepository)
    monkeypatch.setattr(routes, 'UtteranceRepository', FakeUtteranceRepository)
    monkeypatch.setattr(routes, 'SimilaritySearchService', FakeSimilaritySearchService)
    monkeypatch.setattr(routes, 'TwoStageIntentSearchService', FakeTwoStageIntentSearchService)
    monkeypatch.setattr(routes, 'build_reranker_provider', lambda settings: object())
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

    create_utterance = client.post(
        f'/api/v1/intents/{intent_id}/utterances',
        json={'text': 'Kredi karti borcumu ogrenmek istiyorum', 'language_code': 'tr', 'source': 'manual'},
    )
    assert create_utterance.status_code == 200
    utterance_id = create_utterance.json()['id']

    list_utterances = client.get(f'/api/v1/intents/{intent_id}/utterances')
    assert list_utterances.status_code == 200
    assert any(item['id'] == utterance_id for item in list_utterances.json()['items'])

    update_utterance = client.put(
        f'/api/v1/intents/{intent_id}/utterances/{utterance_id}',
        json={'text': 'Kart borcum ne kadar?', 'language_code': 'tr', 'source': 'manual'},
    )
    assert update_utterance.status_code == 200

    search_response = client.post('/api/v1/intents/search', json={'query': 'tarife degistirmek istiyorum', 'k': 3})
    assert search_response.status_code == 200
    assert 'items' in search_response.json()
    assert len(search_response.json()['items']) >= 1

    rerank_search_response = client.post(
        '/api/v1/intents/search/rerank',
        json={'query': 'tarife degistirmek istiyorum', 'k': 3},
    )
    assert rerank_search_response.status_code == 200
    rerank_items = rerank_search_response.json()['items']
    assert len(rerank_items) >= 1
    assert {'intent_code', 'semantic_score', 'reranker_score'} <= set(rerank_items[0].keys())

    delete_utterance = client.delete(f'/api/v1/intents/{intent_id}/utterances/{utterance_id}')
    assert delete_utterance.status_code == 200
    assert delete_utterance.json()['status'] == 'deleted'

    delete_response = client.delete(f'/api/v1/intents/{intent_id}')
    assert delete_response.status_code == 200
    assert delete_response.json()['status'] == 'deleted'
    app.dependency_overrides.clear()


def test_rerank_search_propagates_reranker_error(monkeypatch) -> None:
    class _FakeDb:
        pass

    class _FakeTwoStageIntentSearchService:
        def __init__(self, db, reranker_provider) -> None:
            _ = (db, reranker_provider)

        def top_k(self, query: str, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (query, embedding, k, language_code)
            raise AppError(code='reranker_engine_error', message='Reranker unavailable.', status_code=502)

    monkeypatch.setattr(routes.pipeline.embedding_provider, 'embed', lambda text: EmbeddingResult(vector=_fake_embed(text)))
    monkeypatch.setattr(routes, 'TwoStageIntentSearchService', _FakeTwoStageIntentSearchService)
    monkeypatch.setattr(routes, 'build_reranker_provider', lambda settings: object())
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    client = TestClient(app)

    try:
        response = client.post('/api/v1/intents/search/rerank', json={'query': 'faturami ode', 'k': 3})
        assert response.status_code == 502
        body = response.json()
        assert body['error']['code'] == 'reranker_engine_error'
    finally:
        app.dependency_overrides.clear()
