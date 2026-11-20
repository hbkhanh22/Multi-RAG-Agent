"""Alembic environment: URL and table metadata come from the application itself."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from config.settings import settings
from src.storage.db import metadata, vector_metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

url = settings.sqlalchemy_url
target_metadata = [metadata] + ([vector_metadata] if url.startswith("postgresql") else [])


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=connection.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
