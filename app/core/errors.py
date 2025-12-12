import sys

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from app.core.logging import API_LOGGER
from app.core.exceptions import (
    AppException,
    PromptNotFoundError,
    LLMGenerationError,
    DatabaseError,
    ConfigurationError,
)


async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions."""
    API_LOGGER.error(
        f"Application Error | Type: {type(exc).__name__} | Path: {request.url.path} | Detail: {exc.message}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"An application error occurred: {exc.message}"},
    )

async def prompt_not_found_handler(request: Request, exc: PromptNotFoundError):
    """Handle PromptNotFoundError with a 404 response."""
    API_LOGGER.warning(
        f"Not Found | Path: {request.url.path} | Detail: {exc.message}"
    )
    return JSONResponse(status_code=404, content={"detail": exc.message})

async def llm_generation_handler(request: Request, exc: LLMGenerationError):
    """Handle LLMGenerationError with a 503 response."""
    API_LOGGER.error(
        f"LLM Service Error | Path: {request.url.path} | Detail: {exc.message}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=503, content={"detail": f"LLM service unavailable: {exc.message}"}
    )

async def database_error_handler(request: Request, exc: DatabaseError):
    """Handle DatabaseError with a 500 response."""
    API_LOGGER.error(
        f"Database Error | Path: {request.url.path} | Detail: {exc.message}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500, content={"detail": f"A database error occurred: {exc.message}"}
    )

async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """Handle ConfigurationError with a 500 response."""
    API_LOGGER.error(
        f"Configuration Error | Path: {request.url.path} | Detail: {exc.message}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"A configuration error occurred: {exc.message}"},
    )


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
