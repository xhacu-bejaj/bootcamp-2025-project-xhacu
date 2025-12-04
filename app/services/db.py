from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)


from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

_async_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """
    Initializes and returns the singleton asynchronous SQLAlchemy engine.
    This function handles the configuration once.
    """
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(DATABASE_URL, echo=True, future=True)
    return _async_engine


AsyncSessionLocal = async_sessionmaker(
    bind=get_engine(), class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            # If an exception occurs, roll back the transaction
            await session.rollback()
            raise
        finally:
            await session.close()
