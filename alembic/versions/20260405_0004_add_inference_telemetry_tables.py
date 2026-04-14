"""add inference telemetry tables

Revision ID: 20260405_0004
Revises: 20260402_0003
Create Date: 2026-04-05 16:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '20260405_0004'
down_revision: Union[str, None] = '20260402_0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'inference_stage_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stage_name', sa.String(length=64), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(length=128), nullable=True),
        sa.Column('model_name', sa.String(length=128), nullable=True),
        sa.Column(
            'usage_json',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('error_code', sa.String(length=64), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['request_id'], ['inference_requests.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_inference_stage_metrics_request_id', 'inference_stage_metrics', ['request_id'])
    op.create_index(
        'ix_inference_stage_metrics_stage_name_created_at',
        'inference_stage_metrics',
        ['stage_name', 'created_at'],
    )
    op.create_index('ix_inference_stage_metrics_status_created_at', 'inference_stage_metrics', ['status', 'created_at'])

    op.create_table(
        'inference_cost_pricing',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider', sa.String(length=128), nullable=False),
        sa.Column('model_name', sa.String(length=128), nullable=False),
        sa.Column('unit_type', sa.String(length=32), nullable=False),
        sa.Column('unit_price_usd', sa.Float(), nullable=False),
        sa.Column('effective_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('effective_to', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_inference_cost_pricing_provider_model_unit_effective_from',
        'inference_cost_pricing',
        ['provider', 'model_name', 'unit_type', 'effective_from'],
    )

    op.create_table(
        'inference_metrics_rollup_hourly',
        sa.Column('bucket_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('stage_name', sa.String(length=64), nullable=False),
        sa.Column('request_count', sa.Integer(), nullable=False),
        sa.Column('error_count', sa.Integer(), nullable=False),
        sa.Column('p50_ms', sa.Float(), nullable=False),
        sa.Column('p95_ms', sa.Float(), nullable=False),
        sa.Column('avg_ms', sa.Float(), nullable=False),
        sa.Column('total_estimated_cost_usd', sa.Float(), nullable=False),
        sa.Column('cost_per_1k_requests', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('bucket_start', 'stage_name'),
    )
    op.create_index(
        'ix_inference_metrics_rollup_hourly_stage_name_bucket_start',
        'inference_metrics_rollup_hourly',
        ['stage_name', 'bucket_start'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_inference_metrics_rollup_hourly_stage_name_bucket_start',
        table_name='inference_metrics_rollup_hourly',
    )
    op.drop_table('inference_metrics_rollup_hourly')

    op.drop_index(
        'ix_inference_cost_pricing_provider_model_unit_effective_from',
        table_name='inference_cost_pricing',
    )
    op.drop_table('inference_cost_pricing')

    op.drop_index('ix_inference_stage_metrics_status_created_at', table_name='inference_stage_metrics')
    op.drop_index('ix_inference_stage_metrics_stage_name_created_at', table_name='inference_stage_metrics')
    op.drop_index('ix_inference_stage_metrics_request_id', table_name='inference_stage_metrics')
    op.drop_table('inference_stage_metrics')
