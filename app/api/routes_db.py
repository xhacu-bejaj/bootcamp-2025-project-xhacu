from fastapi import APIRouter, HTTPException, Depends, status

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.services.db import get_session

db_router = APIRouter()


@db_router.get("/health/db", tags=["Database"])
async def db_health_check(session: AsyncSession = Depends(get_session)):
    """
    Checks the database connection by executing a simple 'SELECT 1' query.
    If the query executes successfully, the database connection is healthy.
    """
    try:
        result = await session.execute(text("SELECT 1"))

        if result.scalar_one() == 1:
            return {
                "status": "ok",
                "message": "Database connection is healthy and operational.",
            }
        else:
            raise Exception("Database returned an unexpected result.")

    except Exception as e:
        print(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {e.__class__.__name__}",
        )
