"""Initial schema: conversations, messages, documents, ingest_jobs (+ pgvector chunks)

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-20
"""
import sqlalchemy as sa
from alembic import op

from config.settings import settings

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"
    tz = sa.DateTime(timezone=True)
    # Idempotent: databases created by the app itself (init_db / legacy SQLite) are
    # adopted instead of failing on "table already exists".
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    def create_table(name, *cols, **kw):
        if name not in tables:
            op.create_table(name, *cols, **kw)

    def create_index(name, table, cols, **kw):
        if name not in {i["name"] for i in sa.inspect(bind).get_indexes(table)}:
            op.create_index(name, table, cols, **kw)

    create_table(
        "conversations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=True, index=True),
        sa.Column("title", sa.String(200), nullable=False, server_default=""),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
    )
    create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.String(32),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("meta", sa.JSON, nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("completion_tokens", sa.Integer, nullable=True),
        sa.Column("created_at", sa.String(40), nullable=False),
    )
    create_index("idx_messages_conv", "messages", ["conversation_id", "id"])

    create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("index_name", sa.String(120), nullable=False),
        sa.Column("source_key", sa.String(500), nullable=False),
        sa.Column("domain", sa.String(32), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("size", sa.Integer, nullable=False, server_default="0"),
        sa.Column("mtime", sa.Float, nullable=True),
        sa.Column("etag", sa.String(128), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("chunks_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("created_at", tz, nullable=False),
        sa.Column("updated_at", tz, nullable=False),
        sa.UniqueConstraint("index_name", "source_key", name="uq_documents_index_source"),
    )

    create_table(
        "ingest_jobs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("result", sa.JSON, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", tz, nullable=False),
        sa.Column("started_at", tz, nullable=True),
        sa.Column("finished_at", tz, nullable=True),
    )
    create_index("idx_jobs_status", "ingest_jobs", ["status", "id"])

    if is_pg:
        from pgvector.sqlalchemy import Vector
        from sqlalchemy.dialects.postgresql import JSONB

        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        create_table(
            "chunks",
            sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
            sa.Column("index_name", sa.String(120), nullable=False),
            sa.Column("domain", sa.String(32), nullable=False),
            sa.Column("source_key", sa.String(500), nullable=False),
            sa.Column("filename", sa.String(255), nullable=False),
            sa.Column("content", sa.Text, nullable=False),
            sa.Column("meta", JSONB, nullable=True),
            sa.Column("embedding", Vector(settings.embedding_dimension), nullable=False),
        )
        create_index("idx_chunks_index_domain", "chunks", ["index_name", "domain"])
        create_index("idx_chunks_file", "chunks", ["index_name", "domain", "filename"])
        create_index(
            "idx_chunks_embedding_hnsw", "chunks", ["embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_table("chunks")
    op.drop_table("ingest_jobs")
    op.drop_table("documents")
    op.drop_index("idx_messages_conv", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
