from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware


class APIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        MutableHeaders(scope=request.scope)["X-Request-ID"] = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
