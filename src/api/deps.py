"""FastAPI dependency injection helpers."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session, auto-closing on exit."""
    async with async_session_factory() as session:
        yield session
