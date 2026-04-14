from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.auth import require_api_key
from app.api.errors import AppError
from app.config.settings import get_settings
from app.db.session import get_db
from app.services.audio import AudioValidator
from app.services.confidence import ConfidencePolicy
from app.services.rerank import TwoStageIntentSearchService
from app.services.repository import (
    EmbeddingRepository,
    InferencePricingRepository,
    InferenceRepository,
    InferenceTelemetryRepository,
    IntentRepository,
    UtteranceRepository,
)
from app.services.similarity import SimilaritySearchService
from app.services.storage import LocalStorageProvider
from app.services.telemetry import InferenceTelemetryCollector, estimate_stage_cost
from app.workers.pipeline import (
    InferencePipeline,
    build_embedding_provider,
    build_reranker_provider,
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


class UtteranceWriteRequest(BaseModel):
    text: str = Field(min_length=1)
    language_code: str = Field(default='tr', min_length=2, max_length=8)
    source: str = Field(default='manual', min_length=1, max_length=64)


def _clean_required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise AppError(code='invalid_request', message=f'{field_name} cannot be blank.', status_code=400)
    return cleaned


def _clean_language_code(value: str) -> str:
    cleaned = _clean_required(value, 'language_code').lower()
    if len(cleaned) > 8:
        raise AppError(code='invalid_request', message='language_code is too long.', status_code=400)
    return cleaned


def _parse_iso_timestamp(value: str, field_name: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise AppError(code='invalid_request', message=f'{field_name} is required.', status_code=400)
    normalized = raw[:-1] + '+00:00' if raw.endswith('Z') else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AppError(code='invalid_request', message=f'{field_name} must be valid ISO timestamp.', status_code=400) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_timeframe(start_at: str, end_at: str) -> tuple[datetime, datetime]:
    parsed_start = _parse_iso_timestamp(start_at, 'start_at')
    parsed_end = _parse_iso_timestamp(end_at, 'end_at')
    if parsed_start >= parsed_end:
        raise AppError(code='invalid_request', message='start_at must be earlier than end_at.', status_code=400)
    return parsed_start, parsed_end


def _validate_max_window(start_at: datetime, end_at: datetime, *, max_days: int = 31) -> None:
    if (end_at - start_at).days > max_days:
        raise AppError(
            code='invalid_request',
            message=f'time window cannot exceed {max_days} days.',
            status_code=400,
        )


def _stage_estimated_cost(
    *,
    pricing_repo: InferencePricingRepository,
    provider: str | None,
    model_name: str | None,
    usage: dict[str, int | float | str],
    at: datetime,
) -> float | None:
    if provider is None or model_name is None:
        return None
    try:
        prices = pricing_repo.list_active_prices(provider=provider, model_name=model_name, at=at)
    except SQLAlchemyError:
        return None
    estimated_costs = [
        estimate_stage_cost(
            usage=usage,
            unit_type=price.unit_type,
            unit_price_usd=price.unit_price_usd,
        )
        for price in prices
    ]
    valid_costs = [item for item in estimated_costs if item is not None]
    if not valid_costs:
        return None
    return float(sum(valid_costs))


def _error_code(exc: Exception) -> str:
    if isinstance(exc, AppError):
        return exc.code
    return type(exc).__name__


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
    utterance_repo = UtteranceRepository(db)
    now = datetime.now(timezone.utc).isoformat()
    intent_code = _clean_required(payload.intent_code, 'intent_code')
    description = _clean_required(payload.description, 'description')
    try:
        intent = repo.create(intent_code=intent_code, description=description)
        utterance = utterance_repo.create(
            intent_id=intent.id,
            language_code='tr',
            text=description,
            source='intent_description',
        )
        embedding = pipeline.embedding_provider.embed(utterance.text).vector
        embedding_repo.upsert_for_utterance(
            utterance_id=utterance.id,
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
    utterance_repo = UtteranceRepository(db)
    intent = repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)
    intent_code = _clean_required(payload.intent_code, 'intent_code')
    description = _clean_required(payload.description, 'description')
    try:
        updated = repo.update(intent, intent_code=intent_code, description=description)
        existing_utterances = utterance_repo.by_intent_id(updated.id)
        canonical_utterance = next((item for item in existing_utterances if item.source == 'intent_description'), None)
        if canonical_utterance is None:
            canonical_utterance = utterance_repo.create(
                intent_id=updated.id,
                language_code='tr',
                text=description,
                source='intent_description',
            )
        else:
            utterance_repo.update(
                canonical_utterance,
                language_code='tr',
                text=description,
                source='intent_description',
            )
        embedding = pipeline.embedding_provider.embed(canonical_utterance.text).vector
        embedding_repo.upsert_for_utterance(
            utterance_id=canonical_utterance.id,
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
    utterance_repo = UtteranceRepository(db)
    intents = repo.list_active()
    reindexed_count = 0
    for intent in intents:
        utterances = utterance_repo.by_intent_id(intent.id)
        for utterance in utterances:
            embedding = pipeline.embedding_provider.embed(utterance.text).vector
            embedding_repo.upsert_for_utterance(
                utterance_id=utterance.id,
                model_name=settings.embedding_model_name,
                embedding=embedding,
            )
            reindexed_count += 1
    db.commit()
    return {'status': 'ok', 'reindexed_count': reindexed_count}


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


@router.post('/v1/intents/search/rerank')
def search_intents_rerank(
    payload: IntentSearchRequest,
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, str | float]]]:
    query_text = _clean_required(payload.query, 'query')
    embedding = pipeline.embedding_provider.embed(query_text).vector
    reranker_provider = build_reranker_provider(settings)
    two_stage_search = TwoStageIntentSearchService(db=db, reranker_provider=reranker_provider)
    candidates = two_stage_search.top_k(
        query=query_text,
        embedding=embedding,
        k=payload.k,
        language_code=payload.language_hint,
    )
    return {
        'items': [
            {
                'intent_code': item.intent_code,
                'semantic_score': item.semantic_score,
                'reranker_score': item.reranker_score,
            }
            for item in candidates
        ]
    }


@router.get('/v1/intents/{intent_id}/utterances')
def list_intent_utterances(intent_id: UUID, db: Session = Depends(get_db)) -> dict[str, list[dict[str, str]]]:
    intent_repo = IntentRepository(db)
    utterance_repo = UtteranceRepository(db)
    intent = intent_repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)
    utterances = utterance_repo.by_intent_id(intent_id)
    return {
        'items': [
            {
                'id': str(item.id),
                'intent_id': str(item.intent_id),
                'language_code': item.language_code,
                'text': item.text,
                'source': item.source,
            }
            for item in utterances
        ]
    }


@router.post('/v1/intents/{intent_id}/utterances')
def create_intent_utterance(
    intent_id: UUID,
    payload: UtteranceWriteRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    intent_repo = IntentRepository(db)
    utterance_repo = UtteranceRepository(db)
    embedding_repo = EmbeddingRepository(db)
    intent = intent_repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)

    text = _clean_required(payload.text, 'text')
    language_code = _clean_language_code(payload.language_code)
    source = _clean_required(payload.source, 'source')
    utterance = utterance_repo.create(intent_id=intent.id, language_code=language_code, text=text, source=source)
    embedding = pipeline.embedding_provider.embed(utterance.text).vector
    embedding_repo.upsert_for_utterance(
        utterance_id=utterance.id,
        model_name=settings.embedding_model_name,
        embedding=embedding,
    )
    db.commit()
    return {'id': str(utterance.id), 'intent_id': str(intent.id)}


@router.put('/v1/intents/{intent_id}/utterances/{utterance_id}')
def update_intent_utterance(
    intent_id: UUID,
    utterance_id: UUID,
    payload: UtteranceWriteRequest,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    intent_repo = IntentRepository(db)
    utterance_repo = UtteranceRepository(db)
    embedding_repo = EmbeddingRepository(db)
    intent = intent_repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)
    utterance = utterance_repo.get_by_id(utterance_id)
    if utterance is None or str(utterance.intent_id) != str(intent.id):
        raise AppError(code='utterance_not_found', message='Utterance not found.', status_code=404)

    text = _clean_required(payload.text, 'text')
    language_code = _clean_language_code(payload.language_code)
    source = _clean_required(payload.source, 'source')
    updated = utterance_repo.update(utterance, language_code=language_code, text=text, source=source)
    embedding = pipeline.embedding_provider.embed(updated.text).vector
    embedding_repo.upsert_for_utterance(
        utterance_id=updated.id,
        model_name=settings.embedding_model_name,
        embedding=embedding,
    )
    db.commit()
    return {'id': str(updated.id), 'intent_id': str(intent.id)}


@router.delete('/v1/intents/{intent_id}/utterances/{utterance_id}')
def delete_intent_utterance(intent_id: UUID, utterance_id: UUID, db: Session = Depends(get_db)) -> dict[str, str]:
    intent_repo = IntentRepository(db)
    utterance_repo = UtteranceRepository(db)
    intent = intent_repo.get_by_id(intent_id)
    if intent is None:
        raise AppError(code='intent_not_found', message='Intent not found.', status_code=404)
    utterance = utterance_repo.get_by_id(utterance_id)
    if utterance is None or str(utterance.intent_id) != str(intent.id):
        raise AppError(code='utterance_not_found', message='Utterance not found.', status_code=404)
    utterance_repo.delete(utterance)
    db.commit()
    return {'status': 'deleted', 'id': str(utterance_id)}


@router.post('/v1/inference/intent')
async def infer_intent(
    audio_file: UploadFile = File(...),
    channel_id: str | None = Form(default=None),
    request_id: str | None = Form(default=None),
    language_hint: str = Form(default='tr'),
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> dict:
    repository = InferenceRepository(db)
    pricing_repo = InferencePricingRepository(db)
    telemetry_repo = InferenceTelemetryRepository(db)
    collector = InferenceTelemetryCollector()
    collector.start_stage('total')
    content = await audio_file.read()
    validator.validate(audio_file.filename or 'audio.wav', content)
    audio_uri = storage.store(audio_file.filename or 'audio.wav', content)

    collector.start_stage('stt')
    try:
        transcript_result = pipeline.stt_provider.transcribe(audio_uri=audio_uri, language_code=language_hint)
        stt_usage: dict[str, int | float | str] = {
            'request_count': 1,
            'audio_bytes': len(content),
        }
        collector.end_stage(
            'stt',
            status='ok',
            provider=settings.stt_engine,
            model_name=settings.whisper_model_name,
            usage=stt_usage,
            estimated_cost_usd=_stage_estimated_cost(
                pricing_repo=pricing_repo,
                provider=settings.stt_engine,
                model_name=settings.whisper_model_name,
                usage=stt_usage,
                at=datetime.now(timezone.utc),
            ),
        )
    except Exception as exc:
        collector.end_stage(
            'stt',
            status='error',
            provider=settings.stt_engine,
            model_name=settings.whisper_model_name,
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message=str(exc),
        )
        collector.end_stage(
            'total',
            status='error',
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message='stt_failed',
        )
        raise

    collector.start_stage('embedding')
    try:
        embedding = pipeline.embedding_provider.embed(transcript_result.transcript).vector
        embedding_usage: dict[str, int | float | str] = {
            'request_count': 1,
            'vector_dim': len(embedding),
        }
        collector.end_stage(
            'embedding',
            status='ok',
            provider=settings.embedding_engine,
            model_name=settings.embedding_model_name,
            usage=embedding_usage,
            estimated_cost_usd=_stage_estimated_cost(
                pricing_repo=pricing_repo,
                provider=settings.embedding_engine,
                model_name=settings.embedding_model_name,
                usage=embedding_usage,
                at=datetime.now(timezone.utc),
            ),
        )
    except Exception as exc:
        collector.end_stage(
            'embedding',
            status='error',
            provider=settings.embedding_engine,
            model_name=settings.embedding_model_name,
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message=str(exc),
        )
        collector.end_stage(
            'total',
            status='error',
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message='embedding_failed',
        )
        raise

    similarity = SimilaritySearchService(db)
    collector.start_stage('vector_search')
    try:
        candidates = similarity.top_k(embedding=embedding, k=5, language_code=language_hint)
        vector_usage: dict[str, int | float | str] = {
            'request_count': 1,
            'candidate_count': len(candidates),
        }
        collector.end_stage('vector_search', status='ok', usage=vector_usage)
    except Exception as exc:
        collector.end_stage(
            'vector_search',
            status='error',
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message=str(exc),
        )
        collector.end_stage(
            'total',
            status='error',
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message='vector_search_failed',
        )
        raise

    collector.start_stage('rerank')
    collector.end_stage(
        'rerank',
        status='skipped',
        usage={'request_count': 0, 'reason': 'not_used_in_inference_endpoint'},
    )

    collector.start_stage('confidence_policy')
    try:
        confidence_result = confidence_policy.evaluate(candidates)
        collector.end_stage('confidence_policy', status='ok', usage={'request_count': 1})
    except Exception as exc:
        collector.end_stage(
            'confidence_policy',
            status='error',
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message=str(exc),
        )
        collector.end_stage(
            'total',
            status='error',
            usage={'request_count': 1},
            error_code=_error_code(exc),
            error_message='confidence_policy_failed',
        )
        raise
    match_status = 'low_confidence' if confidence_result.is_low_confidence else 'matched'
    top_intent = confidence_result.top_candidate
    top_candidates = [{'intent_code': candidate.intent_code, 'score': candidate.score} for candidate in confidence_result.top_k]
    effective_request_id = request_id or str(uuid4())
    predicted_intent_id = None
    if match_status == 'matched' and top_intent is not None:
        matched_intent = repository.get_intent_by_code(top_intent.intent_code)
        predicted_intent_id = matched_intent.id if matched_intent is not None else None

    total_ms = 0
    try:
        collector.start_stage('persistence')
        inference_request = repository.create_request(
            external_request_id=effective_request_id,
            language_code=language_hint,
            audio_uri=audio_uri,
            transcript=transcript_result.transcript,
            status=match_status,
            processing_ms=None,
        )
        repository.create_result(
            request_id=inference_request.id,
            predicted_intent_id=predicted_intent_id,
            confidence=confidence_result.confidence,
            top_k_json=top_candidates,
        )
        persistence_usage: dict[str, int | float | str] = {'request_count': 1}
        collector.end_stage(
            'persistence',
            status='ok',
            usage=persistence_usage,
            estimated_cost_usd=_stage_estimated_cost(
                pricing_repo=pricing_repo,
                provider='backend',
                model_name='postgres',
                usage=persistence_usage,
                at=datetime.now(timezone.utc),
            ),
        )
        collector.end_stage('total', status='ok', usage={'request_count': 1})
        total_ms = next(
            (int(item['duration_ms']) for item in collector.stage_rows() if item['stage_name'] == 'total'),
            0,
        )
        inference_request.processing_ms = total_ms
        inference_request.updated_at = datetime.now(timezone.utc)
        db.commit()
        try:
            telemetry_repo.create_stage_metrics(inference_request.id, collector.stage_rows())
            db.commit()
        except SQLAlchemyError:
            db.rollback()
    except IntegrityError as exc:
        try:
            collector.end_stage(
                'persistence',
                status='error',
                usage={'request_count': 1},
                error_code=_error_code(exc),
                error_message='persistence_failed',
            )
        except KeyError:
            pass
        try:
            collector.end_stage(
                'total',
                status='error',
                usage={'request_count': 1},
                error_code=_error_code(exc),
                error_message='persistence_failed',
            )
        except KeyError:
            pass
        db.rollback()
        raise AppError(code='invalid_request', message='Failed to persist inference result.', status_code=400) from exc

    return {
        'request_id': effective_request_id,
        'channel_id': channel_id,
        'intent_code': top_intent.intent_code if top_intent and match_status == 'matched' else None,
        'confidence': confidence_result.confidence,
        'match_status': match_status,
        'transcript': transcript_result.transcript,
        'top_candidates': top_candidates,
        'processing_ms': total_ms,
    }


@router.get('/v1/overview/summary')
def overview_summary(start_at: str, end_at: str, db: Session = Depends(get_db)) -> dict[str, int | float | None]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    repository = InferenceRepository(db)
    return repository.summary(parsed_start, parsed_end)


@router.get('/v1/overview/intent-distribution')
def overview_intent_distribution(start_at: str, end_at: str, db: Session = Depends(get_db)) -> dict[str, list[dict[str, int | float | str]]]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    repository = InferenceRepository(db)
    return {'items': repository.intent_distribution(parsed_start, parsed_end)}


@router.get('/v1/overview/recent-activity')
def overview_recent_activity(
    start_at: str,
    end_at: str,
    limit: int = Query(default=10),
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, object]]]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    if limit < 1 or limit > 50:
        raise AppError(code='invalid_request', message='limit must be between 1 and 50.', status_code=400)
    repository = InferenceRepository(db)
    return {'items': repository.recent_activity(parsed_start, parsed_end, limit)}


@router.get('/v1/overview/stage-latency')
def overview_stage_latency(
    start_at: str,
    end_at: str,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, object]]]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    _validate_max_window(parsed_start, parsed_end)
    repository = InferenceRepository(db)
    return {'items': repository.stage_latency(parsed_start, parsed_end)}


@router.get('/v1/overview/stage-cost')
def overview_stage_cost(
    start_at: str,
    end_at: str,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, object]]]:
    parsed_start, parsed_end = _parse_timeframe(start_at, end_at)
    _validate_max_window(parsed_start, parsed_end)
    repository = InferenceRepository(db)
    return {'items': repository.stage_cost(parsed_start, parsed_end)}


@router.get('/v1/overview/benchmark-compare')
def overview_benchmark_compare(
    baseline_start_at: str,
    baseline_end_at: str,
    candidate_start_at: str,
    candidate_end_at: str,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> dict[str, list[dict[str, object]]]:
    baseline_start, baseline_end = _parse_timeframe(baseline_start_at, baseline_end_at)
    candidate_start, candidate_end = _parse_timeframe(candidate_start_at, candidate_end_at)
    _validate_max_window(baseline_start, baseline_end)
    _validate_max_window(candidate_start, candidate_end)
    repository = InferenceRepository(db)
    return {'items': repository.benchmark_compare(baseline_start, baseline_end, candidate_start, candidate_end)}
