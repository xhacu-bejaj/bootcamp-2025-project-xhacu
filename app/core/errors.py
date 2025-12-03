import sys

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from app.core.logging import API_LOGGER


async def http_exception_handler(request: Request, exc: HTTPException):

    API_LOGGER.warning(
        f"HTTP Client Error | Status: {exc.status_code} | "
        f"Path: {request.url.path} | Detail: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

async def generic_exception_handler(request: Request, exc: Exception):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    
    API_LOGGER.error(
        f"UNHANDLED SERVER ERROR (500) | Path: {request.url.path} | "
        f"Type: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please contact support."}
    )