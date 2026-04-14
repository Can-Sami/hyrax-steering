from app.services.telemetry import InferenceTelemetryCollector, estimate_stage_cost


def test_collector_records_stage_duration_and_status() -> None:
    collector = InferenceTelemetryCollector()
    collector.start_stage('embedding')
    collector.end_stage('embedding', status='ok', provider='openai_compatible', model_name='Qwen')
    rows = collector.stage_rows()

    assert len(rows) == 1
    assert rows[0]['stage_name'] == 'embedding'
    assert rows[0]['status'] == 'ok'
    assert rows[0]['duration_ms'] >= 0


def test_estimate_stage_cost_request_unit() -> None:
    cost = estimate_stage_cost(
        usage={'request_count': 1},
        unit_type='request',
        unit_price_usd=0.0025,
    )
    assert cost == 0.0025


def test_estimate_stage_cost_returns_none_when_usage_missing() -> None:
    cost = estimate_stage_cost(
        usage={},
        unit_type='audio_second',
        unit_price_usd=0.0004,
    )
    assert cost is None
