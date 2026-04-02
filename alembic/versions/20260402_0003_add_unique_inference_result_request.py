"""add unique constraint to inference_results.request_id

Revision ID: 20260402_0003
Revises: 20260401_0002
Create Date: 2026-04-02 14:00:00
"""

from typing import Sequence, Union

from alembic import op

revision: str = '20260402_0003'
down_revision: Union[str, None] = '20260401_0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        'uq_inference_results_request_id',
        'inference_results',
        ['request_id'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_inference_results_request_id',
        'inference_results',
        type_='unique',
    )
