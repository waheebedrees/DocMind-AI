"""Request-correlation middleware.

Every request is assigned an ``X-Request-ID`` — either the value the
client supplied in the same header, or a freshly generated UUID4 — and
that ID is:

* written back onto the request's ASGI scope so downstream middleware,
  dependencies, and endpoint handlers can read it via
  ``request.headers["X-Request-ID"]``;
* echoed on the response so the client can correlate its call with
  server-side logs.

The ID is used as a correlation key in structured logs, not for
authentication or authorization. A client that supplies its own value
is trusted; the middleware does not verify that the value is unique,
well-formed, or that the client has the right to reuse an ID.
"""

from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware


class APIMiddleware(BaseHTTPMiddleware):
    """Propagate an ``X-Request-ID`` across the request/response cycle.

    Positioned at the outermost layer of the middleware stack so that
    every downstream middleware — CORS, error handling, tracing — sees
    the header on the request. If it were placed deeper, a middleware
    above it would log without a correlation ID.

    The middleware does not call into application code; it only
    rewrites headers on the way in and the way out. It cannot fail
    except by failing to allocate a UUID, which is effectively
    impossible.
    """

    async def dispatch(self, request, call_next):
        """Assign or propagate the request's correlation ID.

        Args:
            request: The incoming ``Request``. Its scope's ``headers``
                list is mutated in place — see Notes.
            call_next: The ASGI chain. Called exactly once with the
                mutated scope.

        Returns:
            The response produced by the chain, with ``X-Request-ID``
            set to the same value that was placed on the request.

        Notes:
            ``MutableHeaders(scope=request.scope)`` mutates the ASGI
            header list in place, which is why downstream code sees the
            header even though the request object looks immutable. The
            mutation is invisible to the client but visible to every
            handler reached via ``call_next``.

            The client's ``X-Request-ID`` is trusted verbatim when
            present. It is not length-checked, not character-validated,
            and not verified to be a UUID. If your deployment logs this
            value raw, sanitize it upstream or add validation here —
            see *Known limitations* below.
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        MutableHeaders(scope=request.scope)["X-Request-ID"] = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
