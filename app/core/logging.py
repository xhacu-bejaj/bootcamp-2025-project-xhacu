import logging
import sys
import functools
import os
import time
from typing import Any, Callable

LOG_FILE_PATH = "activity.log"
API_LOGGER: logging.Logger


def setup_logging() -> logging.Logger:
    """Set up logging with file, console, and MongoDB handlers.
    
    Creates a logger with INFO level that writes to:
    - activity.log file
    - stdout console
    - MongoDB logs collection (if connection available)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    global API_LOGGER

    logger = logging.getLogger("LLM_Logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Add MongoDB logging handler
        try:
            from app.core.mongodb_logging import MongoDBHandler
            from app.core.config import settings

            if settings.MONGODB_URI and settings.MONGODB_URI.strip():
                mongodb_handler = MongoDBHandler(settings.MONGODB_URI)
                mongodb_handler.setFormatter(formatter)
                logger.addHandler(mongodb_handler)
        except Exception:
            # Silently skip MongoDB logging if connection fails
            pass

    API_LOGGER = logger
    return logger


API_LOGGER = setup_logging()


import asyncio

def log_api_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log function calls with timing and results.
    
    Logs:
    - Function name and arguments at start
    - Execution latency in milliseconds
    - Return value at completion
    - Exception details if errors occur
    
    Supports both synchronous and asynchronous functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with logging
    """
    os.path.basename(__file__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        start_time = time.perf_counter()

        try:
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            API_LOGGER.info(f"[{func_name}] START Execution. Args: ({signature})")

            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            result_repr = repr(result)
            API_LOGGER.info(
                f"[{func_name}] END Execution. Latency: {latency_ms}ms. Return: {result_repr}"
            )

            return result

        except Exception as e:
            try:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
            except NameError:
                latency_ms = "N/A"

            API_LOGGER.error(
                f"[{func_name}] EXCEPTION raised. Latency: {latency_ms}. "
                f"Type: {type(e).__name__}: {e}",
                exc_info=False,
            )

            raise

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        start_time = time.perf_counter()

        try:
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            API_LOGGER.info(f"[{func_name}] START Execution. Args: ({signature})")

            result = await func(*args, **kwargs)

            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            result_repr = repr(result)
            API_LOGGER.info(
                f"[{func_name}] END Execution. Latency: {latency_ms}ms. Return: {result_repr}"
            )

            return result

        except Exception as e:
            try:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
            except NameError:
                latency_ms = "N/A"

            API_LOGGER.error(
                f"[{func_name}] EXCEPTION raised. Latency: {latency_ms}. "
                f"Type: {type(e).__name__}: {e}",
                exc_info=False,
            )

            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper
