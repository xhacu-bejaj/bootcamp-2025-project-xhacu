import logging
import sys
import functools
import os
import time
from typing import Any, Callable
import traceback

LOG_FILE_PATH = "activity.log"
API_LOGGER: logging.Logger


def setup_logging() -> logging.Logger:

    global API_LOGGER

    logger = logging.getLogger('LLM_Logger')
    logger.setLevel(logging.INFO) 

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )

        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    API_LOGGER = logger 
    return logger

API_LOGGER = setup_logging()


def log_api_call(func: Callable[..., Any]) -> Callable[..., Any]:

    decorator_filename = os.path.basename(__file__)
    
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
            API_LOGGER.info(f"[{func_name}] END Execution. Latency: {latency_ms}ms. Return: {result_repr}")
            
            return result
        
        except Exception as e:
            try:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
            except NameError:
                latency_ms = "N/A"

            exc_type, exc_obj, exc_tb = sys.exc_info()
            location_info = "Location: Unknown"
            
            if exc_tb:
                tb = exc_tb
                while tb.tb_next and os.path.basename(tb.tb_frame.f_code.co_filename) == decorator_filename:
                    tb = tb.tb_next
                
                f = tb.tb_frame
                lineno = tb.tb_lineno
                filename = os.path.split(f.f_code.co_filename)[1]
                location_info = f"Location: {filename}:{lineno}"

            API_LOGGER.error(
                f"[{func_name}] EXCEPTION raised. Latency: {latency_ms}. "
                f"{location_info}. Type: {type(e).__name__}: {e}", 
                exc_info=False
            )
            
            raise

    return wrapper