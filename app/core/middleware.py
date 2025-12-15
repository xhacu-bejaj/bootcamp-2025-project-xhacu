import uuid
from fastapi import Request
from app.core.context import request_id_var


async def add_request_id_middleware(request: Request, call_next):
    """
    Middleware to add a unique request ID to each incoming request.

    The request ID is sourced from the 'X-Request-ID' header if present,
    otherwise a new UUID is generated. The ID is then stored in a
    context variable and added to the response headers.
    """
    # Attempt to get request_id from header, or generate a new one
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    # Store the request_id in the context variable for access in other parts of the app
    request_id_var.set(request_id)
    
    response = await call_next(request)
    
    # Add the request_id to the response headers, so clients can see it
    response.headers["X-Request-ID"] = request_id_var.get()
    
    return response
