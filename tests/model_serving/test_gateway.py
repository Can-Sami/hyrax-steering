from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import httpx
from fastapi.testclient import TestClient


def load_gateway_app(monkeypatch):
    module_name = 'gateway_app_under_test'
    sys.modules.pop(module_name, None)

    fake_fw = types.ModuleType('faster_whisper')

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, *args, **kwargs):
            segment = types.SimpleNamespace(text='ok')
            return [segment], {}

    fake_fw.WhisperModel = DummyWhisperModel
    monkeypatch.setitem(sys.modules, 'faster_whisper', fake_fw)

    app_path = Path(__file__).resolve().parents[2] / 'model-serving' / 'gateway' / 'app.py'
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_embeddings_auth_required(monkeypatch) -> None:
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post('/v1/embeddings', json={'input': 'hello'})

    assert response.status_code == 401


def test_embeddings_requires_input(monkeypatch) -> None:
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post('/v1/embeddings', json={}, headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'})

    assert response.status_code == 400
    assert response.json() == {
        'error': {
            'code': 'missing_required_field',
            'message': 'Request must include required field: input.',
        }
    }


def test_embeddings_accepts_model_field_and_ignores_it_for_cpu_backend(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'transformers_cpu')
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)
    captured = {}

    class CapturingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *args, **kwargs):
            captured['url'] = url
            captured['json'] = kwargs.get('json')
            request = httpx.Request('POST', url)
            return httpx.Response(
                status_code=200,
                request=request,
                json={'data': [{'embedding': [0.1, 0.2]}]},
            )

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: CapturingClient())

    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello', 'model': 'jinaai/jina-embeddings-v3'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 200
    assert 'embeddings-cpu' in captured['url']
    assert captured['json'] == {'input': 'hello', 'model': 'jinaai/jina-embeddings-v3'}


def test_embeddings_rejects_invalid_input_type_with_structured_error(monkeypatch) -> None:
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        '/v1/embeddings',
        json={'input': {'text': 'hello'}},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 400
    body = response.json()
    assert body['error']['code'] == 'invalid_input_type'
    assert body['error']['message'] == 'Input must be a string or list of strings.'


def test_embeddings_rejects_empty_input_string_with_structured_error(monkeypatch) -> None:
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        '/v1/embeddings',
        json={'input': '   '},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 400
    body = response.json()
    assert body['error']['code'] == 'invalid_input'
    assert body['error']['message'] == 'Input string cannot be empty.'


def test_embeddings_upstream_failure_maps_to_502(monkeypatch) -> None:
    module = load_gateway_app(monkeypatch)

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            request = httpx.Request('POST', 'http://embedding-vllm:8000/v1/embeddings')
            response = httpx.Response(status_code=500, request=request)
            raise httpx.HTTPStatusError('upstream failed', request=request, response=response)

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: FailingClient())

    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 502
    assert response.json()['detail'] == 'Embedding upstream failed.'


def test_embeddings_vllm_mode_upstream_failure_returns_503(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'vllm')
    module = load_gateway_app(monkeypatch)

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            request = httpx.Request('POST', 'http://embedding-vllm:8000/v1/embeddings')
            raise httpx.ConnectError('connection failed', request=request)

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: FailingClient())

    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 503


def test_embeddings_invalid_backend_mode_rejected(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'invalid_mode')
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 400
    assert 'EMBEDDING_BACKEND_MODE' in response.json()['detail']


def test_embeddings_transformers_cpu_mode_targets_cpu_service(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'transformers_cpu')
    monkeypatch.setenv('EMBEDDING_CPU_MODEL_NAME', 'jinaai/jina-embeddings-v3')
    module = load_gateway_app(monkeypatch)

    captured = {}

    class CapturingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *args, **kwargs):
            captured['url'] = url
            captured['json'] = kwargs.get('json')
            request = httpx.Request('POST', url)
            return httpx.Response(
                status_code=200,
                request=request,
                json={'data': [{'embedding': [0.1, 0.2]}]},
            )

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: CapturingClient())

    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 200
    assert '/v1/embeddings' in captured['url']
    assert 'embeddings-cpu' in captured['url']
    assert captured['json'] == {'input': 'hello'}
    assert response.json()['model'] == 'jinaai/jina-embeddings-v3'


def test_embeddings_vllm_mode_sends_configured_model(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_BACKEND_MODE', 'vllm')
    monkeypatch.setenv('EMBEDDING_MODEL_NAME', 'Qwen/Qwen3-Embedding-8B')
    module = load_gateway_app(monkeypatch)

    captured = {}

    class CapturingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *args, **kwargs):
            captured['url'] = url
            captured['json'] = kwargs.get('json')
            request = httpx.Request('POST', url)
            return httpx.Response(
                status_code=200,
                request=request,
                json={'data': [{'embedding': [0.1, 0.2]}], 'model': 'Qwen/Qwen3-Embedding-8B'},
            )

    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda *args, **kwargs: CapturingClient())

    client = TestClient(module.app)
    response = client.post(
        '/v1/embeddings',
        json={'input': 'hello'},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 200
    assert captured['json'] == {'input': 'hello', 'model': 'Qwen/Qwen3-Embedding-8B'}
    assert response.json()['model'] == 'Qwen/Qwen3-Embedding-8B'


def test_transcriptions_empty_audio_rejected(monkeypatch) -> None:
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        '/v1/audio/transcriptions',
        data={'model': module.DEFAULT_WHISPER_MODEL, 'language': 'tr'},
        files={'file': ('sample.wav', b'', 'audio/wav')},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 400
    assert response.json()['detail'] == 'Uploaded audio is empty.'


def test_transcriptions_oversized_audio_rejected(monkeypatch) -> None:
    monkeypatch.setenv('MAX_UPLOAD_BYTES', '4')
    module = load_gateway_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        '/v1/audio/transcriptions',
        data={'model': module.DEFAULT_WHISPER_MODEL, 'language': 'tr'},
        files={'file': ('sample.wav', b'12345', 'audio/wav')},
        headers={'Authorization': f'Bearer {module.GATEWAY_API_KEY}'},
    )

    assert response.status_code == 413
    assert response.json()['detail'] == 'Uploaded audio exceeds size limit.'
