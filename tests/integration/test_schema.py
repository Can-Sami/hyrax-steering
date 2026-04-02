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
