"""init schema

Revision ID: 20260331_0001
Revises:
Create Date: 2026-03-31 15:00:00

"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '20260331_0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table(
        'intents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('intent_code', sa.String(length=128), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('intent_code'),
    )

    op.create_table(
        'intent_utterances',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('intent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('language_code', sa.String(length=8), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['intent_id'], ['intents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'intent_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('utterance_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=128), nullable=False),
        sa.Column('embedding', Vector(dim=1024), nullable=False),
        sa.Column('norm', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['utterance_id'], ['intent_utterances.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('utterance_id', 'model_name', name='uq_intent_embeddings_utterance_model'),
    )

    op.create_table(
        'inference_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('external_request_id', sa.String(length=128), nullable=True),
        sa.Column('language_code', sa.String(length=8), nullable=False),
        sa.Column('audio_uri', sa.Text(), nullable=False),
        sa.Column('transcript', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('processing_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('external_request_id', name='uq_inference_requests_external_request_id'),
    )

    op.create_table(
        'inference_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('predicted_intent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('top_k_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('policy_version', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['predicted_intent_id'], ['intents.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['request_id'], ['inference_requests.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'model_registry',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_key', sa.String(length=128), nullable=False),
        sa.Column('version', sa.String(length=64), nullable=False),
        sa.Column('device_mode', sa.String(length=32), nullable=False),
        sa.Column('quantization', sa.String(length=32), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_key', 'version', 'device_mode', name='uq_model_registry_key_version_device'),
    )

    op.create_index('ix_intent_utterances_intent_id', 'intent_utterances', ['intent_id'])
    op.create_index('ix_intent_utterances_language_code', 'intent_utterances', ['language_code'])
    op.create_index('ix_intent_embeddings_utterance_id', 'intent_embeddings', ['utterance_id'])
    op.create_index('ix_inference_requests_created_at', 'inference_requests', ['created_at'])
    op.create_index('ix_inference_requests_language_code', 'inference_requests', ['language_code'])
    op.create_index('ix_inference_results_request_id', 'inference_results', ['request_id'])
    op.create_index('ix_model_registry_model_key', 'model_registry', ['model_key'])


def downgrade() -> None:
    op.drop_index('ix_model_registry_model_key', table_name='model_registry')
    op.drop_index('ix_inference_results_request_id', table_name='inference_results')
    op.drop_index('ix_inference_requests_language_code', table_name='inference_requests')
    op.drop_index('ix_inference_requests_created_at', table_name='inference_requests')
    op.drop_index('ix_intent_embeddings_utterance_id', table_name='intent_embeddings')
    op.drop_index('ix_intent_utterances_language_code', table_name='intent_utterances')
    op.drop_index('ix_intent_utterances_intent_id', table_name='intent_utterances')

    op.drop_table('model_registry')
    op.drop_table('inference_results')
    op.drop_table('inference_requests')
    op.drop_table('intent_embeddings')
    op.drop_table('intent_utterances')
    op.drop_table('intents')
