from app.config.settings import Settings
from app.workers.pipeline import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTranscriptionProvider,
    QwenEmbeddingProvider,
    WhisperLargeV3Provider,
    build_embedding_provider,
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
