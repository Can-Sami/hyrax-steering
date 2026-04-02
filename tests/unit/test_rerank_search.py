from uuid import uuid4

import pytest

from app.api.errors import AppError
from app.domain.schemas import IntentCandidate
from app.services.rerank import TwoStageIntentSearchService
from app.workers.pipeline import OpenAICompatibleRerankerProvider, build_reranker_provider
from app.config.settings import Settings


class _FakeDb:
    pass


def test_openai_compatible_reranker_provider_returns_scores(monkeypatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                'data': [
                    {'index': 0, 'score': 0.11},
                    {'index': 1, 'score': 0.89},
                ]
            }

    captured: dict[str, object] = {}

    def _fake_post(url: str, json: dict, headers: dict, timeout: float):
        captured['url'] = url
        captured['json'] = json
        captured['headers'] = headers
        captured['timeout'] = timeout
        return _FakeResponse()

    monkeypatch.setattr('app.workers.pipeline.httpx.post', _fake_post)

    provider = OpenAICompatibleRerankerProvider(
        base_url='http://localhost:8000/v1',
        api_key='secret',
        model_name='BAAI/bge-reranker-v2-m3',
    )
    scores = provider.score_pairs(
        query='faturami ode',
        documents=['fatura odeme islemleri', 'fatura sorgulama islemleri'],
    )

    assert scores == [0.11, 0.89]
    assert captured['url'] == 'http://localhost:8000/v1/score'
    assert captured['json'] == {
        'model': 'BAAI/bge-reranker-v2-m3',
        'text_1': 'faturami ode',
        'text_2': ['fatura odeme islemleri', 'fatura sorgulama islemleri'],
    }


def test_openai_compatible_reranker_provider_raises_app_error(monkeypatch) -> None:
    class _BadResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {'unexpected': 'shape'}

    monkeypatch.setattr(
        'app.workers.pipeline.httpx.post',
        lambda *args, **kwargs: _BadResponse(),
    )
    provider = OpenAICompatibleRerankerProvider(
        base_url='http://localhost:8000/v1',
        api_key='secret',
        model_name='BAAI/bge-reranker-v2-m3',
    )

    with pytest.raises(AppError) as exc:
        provider.score_pairs(query='q', documents=['d1'])

    assert exc.value.code == 'reranker_engine_error'
    assert exc.value.status_code == 502


def test_openai_compatible_reranker_provider_rejects_invalid_indices(monkeypatch) -> None:
    class _BadIndexResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                'data': [
                    {'index': 1, 'score': 0.55},
                    {'index': 2, 'score': 0.44},
                ]
            }

    monkeypatch.setattr(
        'app.workers.pipeline.httpx.post',
        lambda *args, **kwargs: _BadIndexResponse(),
    )
    provider = OpenAICompatibleRerankerProvider(
        base_url='http://localhost:8000/v1',
        api_key='secret',
        model_name='BAAI/bge-reranker-v2-m3',
    )

    with pytest.raises(AppError) as exc:
        provider.score_pairs(query='q', documents=['d1', 'd2'])

    assert exc.value.code == 'reranker_engine_error'
    assert exc.value.status_code == 502


def test_two_stage_intent_search_reranks_and_keeps_semantic_scores(monkeypatch) -> None:
    intent_a = IntentCandidate(intent_id=uuid4(), intent_code='bill_check', score=0.95)
    intent_b = IntentCandidate(intent_id=uuid4(), intent_code='bill_pay', score=0.91)

    class _FakeSimilarity:
        def __init__(self, db) -> None:
            self.db = db

        def top_k(self, embedding, k: int = 5, language_code: str = 'tr'):
            _ = (embedding, k, language_code)
            return [intent_a, intent_b]

    class _FakeIntentRepo:
        def __init__(self, db) -> None:
            self.db = db

        def get_descriptions_by_ids(self, intent_ids):
            _ = intent_ids
            return {
                intent_a.intent_id: 'fatura sorgulama islemleri',
                intent_b.intent_id: 'fatura odeme islemleri',
            }

    class _FakeReranker:
        def score_pairs(self, query: str, documents: list[str]) -> list[float]:
            _ = (query, documents)
            return [0.10, 0.90]

    monkeypatch.setattr('app.services.rerank.SimilaritySearchService', _FakeSimilarity)
    monkeypatch.setattr('app.services.rerank.IntentRepository', _FakeIntentRepo)

    service = TwoStageIntentSearchService(db=_FakeDb(), reranker_provider=_FakeReranker())
    ranked = service.top_k(query='faturami ode', embedding=[0.1] * 1024, k=2, language_code='tr')

    assert [item.intent_code for item in ranked] == ['bill_pay', 'bill_check']
    assert ranked[0].semantic_score == 0.91
    assert ranked[0].reranker_score == 0.90
    assert ranked[1].semantic_score == 0.95
    assert ranked[1].reranker_score == 0.10


def test_build_reranker_provider_uses_openai_fallback_when_reranker_empty() -> None:
    settings = Settings(
        reranker_engine='openai_compatible',
        reranker_base_url='',
        reranker_api_key='',
        openai_base_url='http://model-gateway:8000/v1',
        openai_api_key='local-dev-key',
        reranker_model_name='BAAI/bge-reranker-v2-m3',
    )
    provider = build_reranker_provider(settings)
    assert isinstance(provider, OpenAICompatibleRerankerProvider)
    assert provider.base_url == 'http://model-gateway:8000/v1'
    assert provider.api_key == 'local-dev-key'
