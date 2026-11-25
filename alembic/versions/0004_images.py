"""Images made by the image agent

Revision ID: 0004_images
Revises: 0003_attachments
Create Date: 2026-11-25
"""
import sqlalchemy as sa
from alembic import op

revision = "0004_images"
down_revision = "0003_attachments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "generated_images" in sa.inspect(op.get_bind()).get_table_names():   # idempotent, like 0001
        return
    op.create_table(
        "generated_images",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=True, index=True),
        sa.Column("conversation_id", sa.String(32), sa.ForeignKey("conversations.id", ondelete="CASCADE"),
                  nullable=True, index=True),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("media_type", sa.String(40), nullable=False),
        sa.Column("data", sa.LargeBinary, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table("generated_images")
