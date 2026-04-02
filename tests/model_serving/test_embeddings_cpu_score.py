import importlib.util
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


def load_embeddings_cpu_app(monkeypatch):
    module_name = 'embeddings_cpu_app_under_test'
    sys.modules.pop(module_name, None)

    fake_hf = types.ModuleType('huggingface_hub')
    fake_hf.hf_hub_download = lambda *args, **kwargs: '/tmp/fake.py'
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hf)

    fake_st = types.ModuleType('sentence_transformers')

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            _ = (convert_to_numpy, normalize_embeddings)
            return [[1.0, 0.0] for _ in texts]

    fake_st.SentenceTransformer = DummySentenceTransformer
    monkeypatch.setitem(sys.modules, 'sentence_transformers', fake_st)

    app_path = Path(__file__).resolve().parents[2] / 'model-serving' / 'embeddings-cpu' / 'app.py'
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_score_returns_openai_compatible_shape(monkeypatch) -> None:
    module = load_embeddings_cpu_app(monkeypatch)

    class DummyModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            _ = (convert_to_numpy, normalize_embeddings)
            if len(texts) == 1:
                return [[1.0, 0.0]]
            return [[0.9, 0.1], [0.2, 0.8]]

    monkeypatch.setattr(module, 'load_model', lambda: DummyModel())
    client = TestClient(module.app)

    response = client.post(
        '/v1/score',
        json={'text_1': 'fatura ode', 'text_2': ['fatura odeme', 'fatura sorgu']},
        headers={'Authorization': f'Bearer {module.api_key}'},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'list'
    assert len(body['data']) == 2
    assert body['data'][0]['index'] == 0
    assert isinstance(body['data'][0]['score'], float)


def test_score_rejects_too_many_documents(monkeypatch) -> None:
    monkeypatch.setenv('EMBEDDING_MAX_SCORE_DOCS', '1')
    module = load_embeddings_cpu_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        '/v1/score',
        json={'text_1': 'q', 'text_2': ['a', 'b']},
        headers={'Authorization': f'Bearer {module.api_key}'},
    )

    assert response.status_code == 413
