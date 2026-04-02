from app.config.settings import Settings
from app.workers.pipeline import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleRerankerProvider,
    OpenAICompatibleTranscriptionProvider,
    QwenEmbeddingProvider,
    WhisperLargeV3Provider,
    build_embedding_provider,
    build_reranker_provider,
    build_stt_provider,
)


def test_provider_factory_uses_stubs_by_default() -> None:
    settings = Settings()
    assert isinstance(build_stt_provider(settings), WhisperLargeV3Provider)
    assert isinstance(build_embedding_provider(settings), QwenEmbeddingProvider)


def test_provider_factory_uses_openai_compatible_engines() -> None:
    settings = Settings(stt_engine='openai_compatible', embedding_engine='openai_compatible')
    assert isinstance(build_stt_provider(settings), OpenAICompatibleTranscriptionProvider)
    assert isinstance(build_embedding_provider(settings), OpenAICompatibleEmbeddingProvider)


def test_provider_factory_builds_openai_compatible_reranker() -> None:
    settings = Settings(
        reranker_engine='openai_compatible',
        reranker_base_url='http://localhost:8001/v1',
        reranker_api_key='key',
        reranker_model_name='BAAI/bge-reranker-v2-m3',
    )
    provider = build_reranker_provider(settings)
    assert isinstance(provider, OpenAICompatibleRerankerProvider)


def test_provider_factory_reranker_falls_back_to_openai_settings() -> None:
    settings = Settings(
        reranker_engine='openai_compatible',
        reranker_base_url='',
        reranker_api_key='',
        reranker_model_name='BAAI/bge-reranker-v2-m3',
        openai_base_url='http://model-gateway:8000/v1',
        openai_api_key='local-dev-key',
    )
    provider = build_reranker_provider(settings)
    assert isinstance(provider, OpenAICompatibleRerankerProvider)
    assert provider.base_url == 'http://model-gateway:8000/v1'
    assert provider.api_key == 'local-dev-key'
