import sys

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from app.core.logging import API_LOGGER


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with logging.
    
    Logs client errors (4xx) as warnings and returns JSON response
    with the original status code and error detail.
    
    Args:
        request: FastAPI request object
        exc: HTTPException raised by the application
        
    Returns:
        JSONResponse with error details
    """
    API_LOGGER.warning(
        f"HTTP Client Error | Status: {exc.status_code} | "
        f"Path: {request.url.path} | Detail: {exc.detail}"
    )

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions as 500 Internal Server Errors.
    
    Logs the full exception with traceback and returns a generic
    500 error response to the client.
    
    Args:
        request: FastAPI request object
        exc: Unhandled exception
        
    Returns:
        JSONResponse with 500 status and generic error message
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()

    API_LOGGER.error(
        f"UNHANDLED SERVER ERROR (500) | Path: {request.url.path} | "
        f"Type: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please contact support."},
    )
