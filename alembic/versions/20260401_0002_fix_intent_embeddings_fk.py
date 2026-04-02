"""fix intent_embeddings foreign key drift

Revision ID: 20260401_0002
Revises: 20260331_0001
Create Date: 2026-04-01 12:30:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '20260401_0002'
down_revision: Union[str, None] = '20260331_0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column['name'] for column in inspector.get_columns('intent_embeddings')}

    # Some local dev databases were initialized with intent_id instead of utterance_id.
    if 'utterance_id' in columns:
        return

    if 'intent_id' not in columns:
        raise RuntimeError("intent_embeddings must contain either 'utterance_id' or 'intent_id'")

    op.add_column('intent_embeddings', sa.Column('utterance_id', sa.UUID(), nullable=True))

    op.execute(
        """
        WITH canonical_utterances AS (
            SELECT DISTINCT ON (intent_id) intent_id, id
            FROM intent_utterances
            ORDER BY intent_id, created_at ASC
        )
        UPDATE intent_embeddings ie
        SET utterance_id = cu.id
        FROM canonical_utterances cu
        WHERE cu.intent_id = ie.intent_id
        """
    )
    op.execute('DELETE FROM intent_embeddings WHERE utterance_id IS NULL')

    op.alter_column('intent_embeddings', 'utterance_id', nullable=False)

    op.drop_constraint('uq_intent_embeddings_intent_model', 'intent_embeddings', type_='unique')
    op.drop_index('ix_intent_embeddings_intent_id', table_name='intent_embeddings')
    op.drop_constraint('intent_embeddings_intent_id_fkey', 'intent_embeddings', type_='foreignkey')
    op.drop_column('intent_embeddings', 'intent_id')

    op.create_foreign_key(
        'intent_embeddings_utterance_id_fkey',
        'intent_embeddings',
        'intent_utterances',
        ['utterance_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_unique_constraint(
        'uq_intent_embeddings_utterance_model',
        'intent_embeddings',
        ['utterance_id', 'model_name'],
    )
    op.create_index('ix_intent_embeddings_utterance_id', 'intent_embeddings', ['utterance_id'])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column['name'] for column in inspector.get_columns('intent_embeddings')}

    if 'intent_id' in columns:
        return
    if 'utterance_id' not in columns:
        raise RuntimeError("intent_embeddings must contain either 'intent_id' or 'utterance_id'")

    op.add_column('intent_embeddings', sa.Column('intent_id', sa.UUID(), nullable=True))

    op.execute(
        """
        UPDATE intent_embeddings ie
        SET intent_id = iu.intent_id
        FROM intent_utterances iu
        WHERE iu.id = ie.utterance_id
        """
    )
    op.execute('DELETE FROM intent_embeddings WHERE intent_id IS NULL')
    op.alter_column('intent_embeddings', 'intent_id', nullable=False)

    op.drop_index('ix_intent_embeddings_utterance_id', table_name='intent_embeddings')
    op.drop_constraint('uq_intent_embeddings_utterance_model', 'intent_embeddings', type_='unique')
    op.drop_constraint('intent_embeddings_utterance_id_fkey', 'intent_embeddings', type_='foreignkey')
    op.drop_column('intent_embeddings', 'utterance_id')

    op.create_foreign_key(
        'intent_embeddings_intent_id_fkey',
        'intent_embeddings',
        'intents',
        ['intent_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_unique_constraint(
        'uq_intent_embeddings_intent_model',
        'intent_embeddings',
        ['intent_id', 'model_name'],
    )
    op.create_index('ix_intent_embeddings_intent_id', 'intent_embeddings', ['intent_id'])
