from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession, 
    expire_on_commit=False 
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