from contextvars import ContextVar
from typing import Optional

# The context variable to store the request ID.
# The default value is None, for code running outside a request context.
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
