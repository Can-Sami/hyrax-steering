from dataclasses import dataclass
from pathlib import Path

import httpx

from app.api.errors import AppError
from app.config.settings import Settings, get_settings


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]


class SttProvider:
    def transcribe(self, audio_uri: str, language_code: str) -> TranscriptionResult:
        raise NotImplementedError


class EmbeddingProvider:
    def embed(self, text: str) -> EmbeddingResult:
        raise NotImplementedError


class WhisperLargeV3Provider(SttProvider):
    def transcribe(self, audio_uri: str, language_code: str) -> TranscriptionResult:
        _ = (audio_uri, language_code)
        return TranscriptionResult(transcript='ornek transcript')


class QwenEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> EmbeddingResult:
        _ = text
        return EmbeddingResult(vector=[0.1] * 1024)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed(self, text: str) -> EmbeddingResult:
        payload = {
            'model': self.settings.embedding_model_name,
            'input': text,
        }
        headers = {'Authorization': f'Bearer {self.settings.openai_api_key}'}

        try:
            response = httpx.post(
                f'{self.settings.openai_base_url}/embeddings',
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            body = response.json()
            vector = body['data'][0]['embedding']
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise AppError(
                code='embedding_engine_error',
                message='Failed to generate embedding from configured engine.',
                status_code=502,
            ) from exc

        return EmbeddingResult(vector=[float(item) for item in vector])


class OpenAICompatibleTranscriptionProvider(SttProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def transcribe(self, audio_uri: str, language_code: str) -> TranscriptionResult:
        audio_path = Path(audio_uri)
        if not audio_path.exists():
            raise AppError(code='audio_not_found', message='Audio file not found for transcription.', status_code=404)

        headers = {'Authorization': f'Bearer {self.settings.openai_api_key}'}
        data = {
            'model': self.settings.whisper_model_name,
            'language': language_code,
        }

        try:
            with audio_path.open('rb') as file_handle:
                files = {'file': (audio_path.name, file_handle, 'application/octet-stream')}
                response = httpx.post(
                    f'{self.settings.openai_base_url}/audio/transcriptions',
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=120.0,
                )
            response.raise_for_status()
            body = response.json()
            transcript = body['text']
        except (httpx.HTTPError, KeyError, TypeError, OSError) as exc:
            raise AppError(
                code='transcription_engine_error',
                message='Failed to transcribe audio from configured engine.',
                status_code=502,
            ) from exc

        return TranscriptionResult(transcript=str(transcript))


def build_stt_provider(settings: Settings) -> SttProvider:
    if settings.stt_engine == 'openai_compatible':
        return OpenAICompatibleTranscriptionProvider(settings)
    return WhisperLargeV3Provider()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_engine == 'openai_compatible':
        return OpenAICompatibleEmbeddingProvider(settings)
    return QwenEmbeddingProvider()


class InferencePipeline:
    def __init__(self, stt_provider: SttProvider, embedding_provider: EmbeddingProvider) -> None:
        self.stt_provider = stt_provider
        self.embedding_provider = embedding_provider
        self.settings = get_settings()

    def run(self, audio_uri: str, language_code: str = 'tr') -> tuple[str, list[float]]:
        transcript = self.stt_provider.transcribe(audio_uri=audio_uri, language_code=language_code)
        embedding = self.embedding_provider.embed(transcript.transcript)
        return transcript.transcript, embedding.vector
