from pathlib import Path

from app.db.base import Base
from app.db import models  # noqa: F401


def test_sqlalchemy_metadata_has_expected_tables() -> None:
    expected_tables = {
        'intents',
        'intent_utterances',
        'intent_embeddings',
        'inference_requests',
        'inference_results',
        'model_registry',
        'inference_stage_metrics',
        'inference_cost_pricing',
        'inference_metrics_rollup_hourly',
    }
    assert expected_tables.issubset(set(Base.metadata.tables.keys()))


def test_initial_migration_defines_pgvector_extension_and_core_tables() -> None:
    migration_file = Path('alembic/versions/20260331_0001_init_schema.py')
    content = migration_file.read_text(encoding='utf-8')

    assert "CREATE EXTENSION IF NOT EXISTS vector" in content
    assert "op.create_table(\n        'intents'" in content
    assert "op.create_table(\n        'intent_embeddings'" in content
    assert "sa.Column('description', sa.Text(), nullable=False)" in content
    assert "sa.Column('utterance_id', postgresql.UUID(as_uuid=True), nullable=False)" in content
    assert "uq_intent_embeddings_utterance_model" in content
    assert "Vector(dim=1024)" in content


def test_latest_migration_adds_unique_constraint_for_inference_result_request() -> None:
    migration_file = Path('alembic/versions/20260402_0003_add_unique_inference_result_request.py')
    content = migration_file.read_text(encoding='utf-8')
    assert "op.create_unique_constraint(\n        'uq_inference_results_request_id'" in content


def test_telemetry_migration_defines_stage_metric_and_pricing_tables() -> None:
    migration_file = Path('alembic/versions/20260405_0004_add_inference_telemetry_tables.py')
    content = migration_file.read_text(encoding='utf-8')
    assert "op.create_table(\n        'inference_stage_metrics'" in content
    assert "op.create_table(\n        'inference_cost_pricing'" in content
    assert "op.create_table(\n        'inference_metrics_rollup_hourly'" in content
