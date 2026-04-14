from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


client = TestClient(app)


class _FakeDb:
    pass


def test_overview_summary_returns_metrics(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def summary(self, start_at, end_at):
            assert start_at.tzinfo is not None
            assert end_at.tzinfo is not None
            return {
                'total_inferences': 10,
                'avg_confidence': 0.75,
                'total_inferences_delta_pct': 25.0,
                'avg_confidence_delta_pct': -5.0,
            }

    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    try:
        response = client.get(
            '/api/v1/overview/summary',
            params={
                'start_at': '2026-04-01T10:00:00Z',
                'end_at': '2026-04-01T12:00:00Z',
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            'total_inferences': 10,
            'avg_confidence': 0.75,
            'total_inferences_delta_pct': 25.0,
            'avg_confidence_delta_pct': -5.0,
        }
    finally:
        app.dependency_overrides.clear()


def test_overview_intent_distribution_includes_unmatched(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def intent_distribution(self, start_at, end_at):
            _ = (start_at, end_at)
            return [
                {'intent_code': 'billing', 'count': 7, 'percentage': 70.0},
                {'intent_code': 'unmatched', 'count': 3, 'percentage': 30.0},
            ]

    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    try:
        response = client.get(
            '/api/v1/overview/intent-distribution',
            params={
                'start_at': '2026-04-01T10:00:00+00:00',
                'end_at': '2026-04-01T12:00:00+00:00',
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload['items'][0]['count'] >= payload['items'][1]['count']
        assert payload['items'][1]['intent_code'] == 'unmatched'
    finally:
        app.dependency_overrides.clear()


def test_overview_recent_activity_respects_limit(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def recent_activity(self, start_at, end_at, limit: int):
            _ = (start_at, end_at)
            all_items = [
                {
                    'timestamp': datetime(2026, 4, 1, 11, 59, tzinfo=timezone.utc).isoformat(),
                    'input_snippet': 'newest',
                    'predicted_intent': 'billing',
                    'confidence': 0.9,
                },
                {
                    'timestamp': datetime(2026, 4, 1, 11, 58, tzinfo=timezone.utc).isoformat(),
                    'input_snippet': 'older',
                    'predicted_intent': None,
                    'confidence': 0.4,
                },
            ]
            return all_items[:limit]

    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    try:
        response = client.get(
            '/api/v1/overview/recent-activity',
            params={
                'start_at': '2026-04-01T10:00:00Z',
                'end_at': '2026-04-01T12:00:00Z',
                'limit': 1,
            },
        )
        assert response.status_code == 200
        payload = response.json()['items']
        assert len(payload) == 1
        assert payload[0]['input_snippet'] == 'newest'
    finally:
        app.dependency_overrides.clear()


def test_overview_invalid_timeframe_returns_400() -> None:
    response = client.get(
        '/api/v1/overview/summary',
        params={
            'start_at': '2026-04-01T12:00:00Z',
            'end_at': '2026-04-01T10:00:00Z',
        },
    )
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'invalid_request'


def test_overview_recent_activity_limit_bounds() -> None:
    response = client.get(
        '/api/v1/overview/recent-activity',
        params={
            'start_at': '2026-04-01T10:00:00Z',
            'end_at': '2026-04-01T12:00:00Z',
            'limit': 51,
        },
    )
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'invalid_request'


def test_overview_stage_latency_returns_rollup(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def stage_latency(self, start_at, end_at):
            _ = (start_at, end_at)
            return [
                {
                    'stage_name': 'embedding',
                    'request_count': 12,
                    'error_count': 0,
                    'p50_ms': 23.0,
                    'p95_ms': 41.0,
                    'avg_ms': 25.5,
                }
            ]

    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    try:
        response = client.get(
            '/api/v1/overview/stage-latency',
            params={
                'start_at': '2026-04-01T10:00:00Z',
                'end_at': '2026-04-01T12:00:00Z',
            },
        )
        assert response.status_code == 200
        payload = response.json()['items']
        assert payload[0]['stage_name'] == 'embedding'
        assert payload[0]['p95_ms'] == 41.0
    finally:
        app.dependency_overrides.clear()


def test_overview_stage_cost_returns_rollup(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def stage_cost(self, start_at, end_at):
            _ = (start_at, end_at)
            return [
                {
                    'stage_name': 'stt',
                    'request_count': 20,
                    'total_estimated_cost_usd': 0.3,
                    'cost_per_1k_requests': 15.0,
                }
            ]

    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    try:
        response = client.get(
            '/api/v1/overview/stage-cost',
            params={
                'start_at': '2026-04-01T10:00:00Z',
                'end_at': '2026-04-01T12:00:00Z',
            },
        )
        assert response.status_code == 200
        payload = response.json()['items']
        assert payload[0]['stage_name'] == 'stt'
        assert payload[0]['cost_per_1k_requests'] == 15.0
    finally:
        app.dependency_overrides.clear()


def test_overview_benchmark_compare_returns_deltas(monkeypatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, db) -> None:
            self.db = db

        def benchmark_compare(self, baseline_start, baseline_end, candidate_start, candidate_end):
            _ = (baseline_start, baseline_end, candidate_start, candidate_end)
            return [
                {
                    'stage_name': 'vector_search',
                    'baseline_p95_ms': 50.0,
                    'candidate_p95_ms': 42.0,
                    'p95_delta_pct': -16.0,
                    'baseline_cost_per_1k_requests': 10.0,
                    'candidate_cost_per_1k_requests': 8.0,
                    'cost_per_1k_delta_pct': -20.0,
                }
            ]

    monkeypatch.setattr(routes, 'InferenceRepository', _FakeInferenceRepository)
    app.dependency_overrides[routes.get_db] = lambda: _FakeDb()
    try:
        response = client.get(
            '/api/v1/overview/benchmark-compare',
            params={
                'baseline_start_at': '2026-04-01T08:00:00Z',
                'baseline_end_at': '2026-04-01T10:00:00Z',
                'candidate_start_at': '2026-04-01T10:00:00Z',
                'candidate_end_at': '2026-04-01T12:00:00Z',
            },
        )
        assert response.status_code == 200
        payload = response.json()['items']
        assert payload[0]['stage_name'] == 'vector_search'
        assert payload[0]['p95_delta_pct'] == -16.0
    finally:
        app.dependency_overrides.clear()
