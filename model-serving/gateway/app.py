from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# Import get_embedding_backend_url from embedding_router.py
_embedding_router_path = Path(__file__).parent / 'embedding_router.py'
_embedding_router_spec = importlib.util.spec_from_file_location('embedding_router', _embedding_router_path)
_embedding_router = importlib.util.module_from_spec(_embedding_router_spec)
sys.modules['embedding_router'] = _embedding_router
_embedding_router_spec.loader.exec_module(_embedding_router)
get_embedding_backend_url = _embedding_router.get_embedding_backend_url

app = FastAPI(title='callsteering-model-gateway', version='0.1.0')

APP_ENV = os.getenv('APP_ENV', 'dev')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY', 'local-dev-key')
GATEWAY_API_KEY = os.getenv('GATEWAY_API_KEY', 'local-dev-key')
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME', 'Qwen/Qwen3-Embedding-8B')
DEFAULT_WHISPER_MODEL = os.getenv('WHISPER_MODEL_NAME', 'Qwen/Qwen3-ASR-1.7B')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
QWEN_ASR_DEVICE_MAP = os.getenv('QWEN_ASR_DEVICE_MAP', 'cpu')
QWEN_ASR_DTYPE = os.getenv('QWEN_ASR_DTYPE', 'float32')
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(25 * 1024 * 1024)))
UPLOAD_CHUNK_SIZE = int(os.getenv('UPLOAD_CHUNK_SIZE', str(1024 * 1024)))
MAX_SCORE_QUERY_CHARS = int(os.getenv('MAX_SCORE_QUERY_CHARS', '4096'))
MAX_SCORE_DOCS = int(os.getenv('MAX_SCORE_DOCS', '64'))
MAX_SCORE_DOC_CHARS = int(os.getenv('MAX_SCORE_DOC_CHARS', '4096'))
MAX_SCORE_TOTAL_CHARS = int(os.getenv('MAX_SCORE_TOTAL_CHARS', '32768'))

WHISPER_MODEL_ALIASES: dict[str, str] = {
    'openai/whisper-large-v3': 'large-v3',
    'whisper-large-v3': 'large-v3',
}


_whisper: WhisperModel | None = None
_qwen_asr_model: Any | None = None


def is_default_key(value: str) -> bool:
    return value.strip() in {'', 'local-dev-key'}


@app.on_event('startup')
def validate_security_config() -> None:
    if APP_ENV.lower() in {'prod', 'production'}:
        if is_default_key(GATEWAY_API_KEY) or is_default_key(EMBEDDING_API_KEY):
            raise RuntimeError('Default API keys are not allowed in production mode.')


def resolve_whisper_runtime_model(model_name: str) -> str:
    return WHISPER_MODEL_ALIASES.get(model_name, model_name)


def get_whisper_model() -> WhisperModel:
    global _whisper
    if _whisper is None:
        runtime_model = resolve_whisper_runtime_model(DEFAULT_WHISPER_MODEL)
        _whisper = WhisperModel(runtime_model, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _whisper


def is_qwen_asr_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith('qwen/qwen3-asr')


def get_qwen_asr_model() -> Any:
    global _qwen_asr_model
    if _qwen_asr_model is None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError('qwen_asr package is required for Qwen3-ASR models.') from exc

        _qwen_asr_model = Qwen3ASRModel.from_pretrained(
            DEFAULT_WHISPER_MODEL,
            device_map=QWEN_ASR_DEVICE_MAP,
            dtype=QWEN_ASR_DTYPE,
        )
    return _qwen_asr_model


def get_backend_model_name(backend_name: str) -> str:
    if backend_name == 'transformers_cpu':
        return os.getenv('EMBEDDING_CPU_MODEL_NAME', 'Qwen/Qwen3-Embedding-4B')
    return DEFAULT_EMBEDDING_MODEL


def get_vllm_base_url() -> str:
    return os.getenv('EMBEDDING_VLLM_BASE_URL', 'http://embedding-vllm:8000/v1').rstrip('/')


def get_cpu_base_url() -> str:
    return os.getenv('EMBEDDING_CPU_BASE_URL', 'http://embeddings-cpu:8000/v1').rstrip('/')


def check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing bearer token.')
    token = authorization.split(' ', 1)[1]
    if token != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail='Invalid token.')


def bad_request(code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            'error': {
                'code': code,
                'message': message,
            }
        },
    )


@app.get('/healthz')
def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/readyz')
def readyz(backend: str | None = Query(default=None)) -> dict[str, str]:
    result = {'status': 'ready', 'service': 'callsteering-model-gateway'}
    
    if backend is not None:
        backend = backend.strip().lower()
        if backend not in {'vllm', 'transformers_cpu'}:
            raise HTTPException(status_code=400, detail=f'Invalid backend: {backend!r}. Must be "vllm" or "transformers_cpu".')
    
    try:
        mode, _ = get_embedding_backend_url()
        result['embedding_mode'] = mode
    except ValueError:
        pass
    
    return result


@app.post('/v1/embeddings')
async def embeddings(payload: dict, authorization: str | None = Header(default=None)) -> JSONResponse:
    check_auth(authorization)

    try:
        backend_name, backend_url = get_embedding_backend_url()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    allowed_fields = {'input', 'model'}
    unknown_fields = sorted(field for field in payload.keys() if field not in allowed_fields)
    if unknown_fields:
        return bad_request(
            code='invalid_request',
            message='Invalid request payload.',
        )

    if 'input' not in payload:
        return bad_request(
            code='missing_required_field',
            message='Request must include required field: input.',
        )

    input_value = payload.get('input')
    if isinstance(input_value, str):
        if not input_value.strip():
            return bad_request(
                code='invalid_input',
                message='Input string cannot be empty.',
            )
    elif isinstance(input_value, list):
        if not input_value:
            return bad_request(
                code='invalid_input',
                message='Input list cannot be empty.',
            )
        for index, item in enumerate(input_value):
            if not isinstance(item, str) or not item.strip():
                return bad_request(
                    code='invalid_input',
                    message='Input list must contain non-empty strings only.',
                )
    else:
        return bad_request(
            code='invalid_input_type',
            message='Input must be a string or list of strings.',
        )

    request_body = {'input': input_value}
    if backend_name == 'vllm':
        request_body['model'] = DEFAULT_EMBEDDING_MODEL
    elif backend_name == 'transformers_cpu':
        # CPU backend exposes OpenAI-compatible /v1/embeddings and accepts optional model.
        # Forwarding preserves compatibility with upstream callers that always include model.
        if 'model' in payload:
            request_body['model'] = payload['model']
    headers = {'Authorization': f'Bearer {EMBEDDING_API_KEY}'}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f'{backend_url}/embeddings', json=request_body, headers=headers)
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise HTTPException(status_code=503, detail='Embedding upstream unavailable.') from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail='Embedding upstream failed.') from exc

    response_data = response.json()
    response_data['model'] = response_data.get('model') or get_backend_model_name(backend_name)
    response_data['backend_used'] = backend_name
    return JSONResponse(response_data)


@app.post('/v1/audio/transcriptions')
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_WHISPER_MODEL),
    language: str | None = Form(default='tr'),
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    check_auth(authorization)

    suffix = Path(file.filename or 'audio.wav').suffix or '.wav'
    temp_path = None
    bytes_written = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail='Uploaded audio exceeds size limit.')
                temp_file.write(chunk)

            if bytes_written == 0:
                raise HTTPException(status_code=400, detail='Uploaded audio is empty.')

        model_name = model if model else DEFAULT_WHISPER_MODEL
        if model_name != DEFAULT_WHISPER_MODEL:
            raise HTTPException(status_code=400, detail=f'Only model {DEFAULT_WHISPER_MODEL} is supported by this gateway.')

        if is_qwen_asr_model(model_name):
            qwen_model = get_qwen_asr_model()
            results = qwen_model.transcribe(audio=temp_path, language=_qwen_language(language))
            if not results:
                raise RuntimeError('Qwen ASR returned no transcription results.')
            text = str(getattr(results[0], 'text', '')).strip()
        else:
            whisper = get_whisper_model()
            segments, _ = whisper.transcribe(temp_path, language=language)
            text = ' '.join(segment.text.strip() for segment in segments).strip()
        return {'text': text}
    except OSError as exc:
        raise HTTPException(status_code=500, detail='Could not process uploaded audio.') from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail='Transcription engine failed to load or run.') from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized:
        return None
    mapping = {
        'tr': 'Turkish',
        'en': 'English',
        'zh': 'Chinese',
        'yue': 'Cantonese',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
    }
    return mapping.get(normalized, language)


@app.post('/v1/score')
async def score(payload: dict, authorization: str | None = Header(default=None)) -> JSONResponse:
    check_auth(authorization)
    allowed_fields = {'model', 'text_1', 'text_2'}
    unknown_fields = sorted(field for field in payload.keys() if field not in allowed_fields)
    if unknown_fields:
        return bad_request(
            code='invalid_request',
            message='Invalid request payload.',
        )
    if 'text_1' not in payload or 'text_2' not in payload:
        return bad_request(
            code='missing_required_field',
            message='Request must include required fields: text_1, text_2.',
        )
    if not isinstance(payload['text_1'], str) or not payload['text_1'].strip():
        return bad_request(code='invalid_input', message='text_1 must be a non-empty string.')
    if not isinstance(payload['text_2'], list) or not payload['text_2']:
        return bad_request(code='invalid_input', message='text_2 must be a non-empty list of strings.')
    if len(payload['text_2']) > MAX_SCORE_DOCS:
        return bad_request(code='invalid_input', message='text_2 exceeds max document count.')
    if not all(isinstance(item, str) and item.strip() for item in payload['text_2']):
        return bad_request(code='invalid_input', message='text_2 must contain non-empty strings only.')
    if len(payload['text_1']) > MAX_SCORE_QUERY_CHARS:
        return bad_request(code='invalid_input', message='text_1 exceeds max length.')
    if any(len(item) > MAX_SCORE_DOC_CHARS for item in payload['text_2']):
        return bad_request(code='invalid_input', message='text_2 contains documents exceeding max length.')
    total_chars = len(payload['text_1']) + sum(len(item) for item in payload['text_2'])
    if total_chars > MAX_SCORE_TOTAL_CHARS:
        return bad_request(code='invalid_input', message='score request exceeds total character limit.')

    request_body = {
        'model': payload.get('model') or os.getenv('RERANKER_MODEL_NAME', 'BAAI/bge-reranker-v2-m3'),
        'text_1': payload['text_1'],
        'text_2': payload['text_2'],
    }
    headers = {'Authorization': f'Bearer {EMBEDDING_API_KEY}'}
    backend_name, _ = get_embedding_backend_url()
    score_base_url = get_vllm_base_url() if backend_name == 'vllm' else get_cpu_base_url()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f'{score_base_url}/score', json=request_body, headers=headers)
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise HTTPException(status_code=503, detail='Reranker upstream unavailable.') from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail='Reranker upstream failed.') from exc

    return JSONResponse(response.json())
