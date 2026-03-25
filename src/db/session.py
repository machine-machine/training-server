"""Async database session factory."""

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://dexy_training:password@postgres:5432/dexy_training",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create tables if they don't exist (Alembic handles migrations in prod)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
