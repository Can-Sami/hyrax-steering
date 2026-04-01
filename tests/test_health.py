from fastapi.testclient import TestClient

from app.main import app


def test_healthz_has_request_id() -> None:
    client = TestClient(app)
    response = client.get('/api/healthz')

    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.headers['x-request-id']


def test_readyz_has_service_metadata() -> None:
    client = TestClient(app)
    response = client.get('/api/readyz', headers={'x-request-id': 'req-123'})

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'
    assert body['service'] == 'callsteering-backend'
    assert body['request_id'] == 'req-123'
    assert response.headers['x-request-id'] == 'req-123'
