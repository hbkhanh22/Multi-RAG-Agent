"""Authentication: users and refresh_tokens

Revision ID: 0002_auth
Revises: 0001_initial
Create Date: 2026-11-20
"""
import sqlalchemy as sa
from alembic import op

revision = "0002_auth"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())   # idempotent, like 0001
    tz = sa.DateTime(timezone=True)

    if "users" not in tables:
        op.create_table(
            "users",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("email", sa.String(320), nullable=False, unique=True),
            sa.Column("name", sa.String(200), nullable=False, server_default=""),
            sa.Column("password_hash", sa.String(300), nullable=True),
            sa.Column("google_sub", sa.String(64), nullable=True, unique=True),
            sa.Column("role", sa.String(16), nullable=False, server_default="user"),
            sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
            sa.Column("created_at", tz, nullable=False),
            sa.Column("last_login_at", tz, nullable=True),
        )
    if "refresh_tokens" not in tables:
        op.create_table(
            "refresh_tokens",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"),
                      nullable=False, index=True),
            sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
            sa.Column("family", sa.String(32), nullable=False),
            sa.Column("expires_at", tz, nullable=False),
            sa.Column("revoked_at", tz, nullable=True),
            sa.Column("created_at", tz, nullable=False),
            sa.Column("user_agent", sa.String(300), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("refresh_tokens")
    op.drop_table("users")
