from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import require_api_key
from app.api.errors import AppError
from app.config.settings import get_settings
from app.db.session import get_db
from app.services.audio import AudioValidator
from app.services.confidence import ConfidencePolicy
from app.services.repository import EmbeddingRepository, IntentRepository
from app.services.similarity import SimilaritySearchService
from app.services.storage import LocalStorageProvider
from app.workers.pipeline import (
    InferencePipeline,
    build_embedding_provider,
    build_stt_provider,
)

router = APIRouter()
validator = AudioValidator()
storage = LocalStorageProvider()
confidence_policy = ConfidencePolicy()
settings = get_settings()
pipeline = InferencePipeline(
    stt_provider=build_stt_provider(settings),
    embedding_provider=build_embedding_provider(settings),
)


class IntentWriteRequest(BaseModel):
    intent_code: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1)


class IntentSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=25)
    language_hint: str = Field(default='tr')


def _clean_required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise AppError(code='invalid_request', message=f'{field_name} cannot be blank.', status_code=400)
    return cleaned


@router.get('/healthz')
def healthz(request: Request) -> dict[str, str]:
    return {
        'status': 'ok',
        'request_id': request.state.request_id,
    }


@router.get('/readyz')
def readyz(request: Request) -> dict[str, str]:
    settings = get_settings()
    return {
        'status': 'ready',
        'service': settings.app_name,
        'version': settings.app_version,
        'request_id': request.state.request_id,
    }


@router.get('/v1/intents')
def list_intents(db: Session = Depends(get_db)) -> dict[str, list[dict[str, str | bool]]]:
    repo = IntentRepository(db)
    intents = repo.list_all()
    return {
        'items': [
            {
                'id': str(intent.id),
                'intent_code': intent.intent_code,
                'description': intent.description,
                'is_active': intent.is_active,
            }
            for intent in intents
        ]
    }


@router.post('/v1/intents')
def create_intent(payload: IntentWriteRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    repo = IntentRepository(db)
    embedding_repo = EmbeddingRepository(db)
    now = datetime.now(timezone.utc).isoformat()
    intent_code = _clean_required(payload.intent_code, 'intent_code')
    description = _clean_required(payload.description, 'description')
    try:
        intent = repo.create(intent_code=intent_code, description=description)
        text = f'{intent.intent_code}\n{intent.description}'
        embedding = pipeline.embedding_provider.embed(text).vector
        embedding_repo.upsert_for_intent(
            intent_id=intent.id,
            model_name=settings.embedding_model_name,
            embedding=embedding,
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise AppError(
            code='intent_conflict',
            message='Intent code already exists.',
            status_code=409,
        ) from exc
    return {'id': str(intent.id), 'intent_code': intent.intent_code, 'created_at': now}


@router.put('/v1/intents/{intent_id}')
def update_intent(intent_id: UUID, payload: IntentWriteRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    repo = IntentRepository(db)
    embedding_repo = EmbeddingRepository(db)
    intent = repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)
    intent_code = _clean_required(payload.intent_code, 'intent_code')
    description = _clean_required(payload.description, 'description')
    try:
        updated = repo.update(intent, intent_code=intent_code, description=description)
        text = f'{updated.intent_code}\n{updated.description}'
        embedding = pipeline.embedding_provider.embed(text).vector
        embedding_repo.upsert_for_intent(
            intent_id=updated.id,
            model_name=settings.embedding_model_name,
            embedding=embedding,
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise AppError(
            code='intent_conflict',
            message='Intent code already exists.',
            status_code=409,
        ) from exc
    return {'id': str(updated.id), 'intent_code': updated.intent_code}


@router.delete('/v1/intents/{intent_id}')
def delete_intent(intent_id: UUID, db: Session = Depends(get_db)) -> dict[str, str]:
    repo = IntentRepository(db)
    intent = repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)
    repo.delete(intent)
    db.commit()
    return {'status': 'deleted', 'id': str(intent_id)}


@router.post('/v1/intents/reindex')
def reindex_intents(db: Session = Depends(get_db)) -> dict[str, str | int]:
    repo = IntentRepository(db)
    embedding_repo = EmbeddingRepository(db)
    intents = repo.list_active()
    for intent in intents:
        text = f'{intent.intent_code}\n{intent.description}'
        embedding = pipeline.embedding_provider.embed(text).vector
        embedding_repo.upsert_for_intent(intent_id=intent.id, model_name=settings.embedding_model_name, embedding=embedding)
    db.commit()
    return {'status': 'ok', 'reindexed_count': len(intents)}


@router.post('/v1/intents/search')
def search_intents(payload: IntentSearchRequest, db: Session = Depends(get_db)) -> dict[str, list[dict[str, str | float]]]:
    query_text = _clean_required(payload.query, 'query')
    embedding = pipeline.embedding_provider.embed(query_text).vector
    similarity = SimilaritySearchService(db)
    candidates = similarity.top_k(embedding=embedding, k=payload.k, language_code=payload.language_hint)
    return {
        'items': [
            {
                'intent_code': item.intent_code,
                'score': item.score,
            }
            for item in candidates
        ]
    }


@router.post('/v1/inference/intent')
async def infer_intent(
    audio_file: UploadFile = File(...),
    channel_id: str | None = Form(default=None),
    request_id: str | None = Form(default=None),
    language_hint: str = Form(default='tr'),
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> dict:
    content = await audio_file.read()
    validator.validate(audio_file.filename or 'audio.wav', content)
    audio_uri = storage.store(audio_file.filename or 'audio.wav', content)

    transcript, embedding = pipeline.run(audio_uri=audio_uri, language_code=language_hint)
    similarity = SimilaritySearchService(db)
    candidates = similarity.top_k(embedding=embedding, k=5, language_code=language_hint)
    confidence_result = confidence_policy.evaluate(candidates)
    match_status = 'low_confidence' if confidence_result.is_low_confidence else 'matched'
    top_intent = confidence_result.top_candidate

    return {
        'request_id': request_id or str(uuid4()),
        'channel_id': channel_id,
        'intent_code': top_intent.intent_code if top_intent and match_status == 'matched' else None,
        'confidence': confidence_result.confidence,
        'match_status': match_status,
        'transcript': transcript,
        'top_candidates': [
            {'intent_code': candidate.intent_code, 'score': candidate.score}
            for candidate in confidence_result.top_k
        ],
        'processing_ms': 120,
    }
