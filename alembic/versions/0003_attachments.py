"""Files attached to chat messages

Revision ID: 0003_attachments
Revises: 0002_auth
Create Date: 2026-11-25
"""
import sqlalchemy as sa
from alembic import op

revision = "0003_attachments"
down_revision = "0002_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "attachments" in sa.inspect(op.get_bind()).get_table_names():   # idempotent, like 0001
        return
    op.create_table(
        "attachments",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("conversation_id", sa.String(32), sa.ForeignKey("conversations.id", ondelete="CASCADE"),
                  nullable=True, index=True),
        sa.Column("filename", sa.String(200), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("pages", sa.Integer, nullable=True),
        sa.Column("chars", sa.Integer, nullable=False),
        sa.Column("segments", sa.JSON, nullable=False),
        sa.Column("image", sa.LargeBinary, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table("attachments")
