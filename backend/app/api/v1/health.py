"""Health and readiness endpoint.

Reports whether the service is ready to serve traffic by probing its
two hard dependencies, Postgres and Redis. The response is a structured
object, not a bare string, so a load balancer or orchestrator can
distinguish "postgres is down but redis is fine" from "everything is
down" without parsing logs.

The endpoint returns 200 when both dependencies are reachable and 503
when either is not. It does *not* return 503 while a dependency is
still warming up after a restart — a dependency that hasn't finished
connecting is treated the same as one that's broken.
"""

import time
from typing import Literal

from asyncpg import PostgresError
from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from redis import AuthenticationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import DbSession
from app.core.logging import get_logger
from app.db.redis import get_redis

log = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# "error" is intentionally the only failure state. Adding "degraded" or
# "unknown" would force consumers to decide which states they should
# serve traffic for, which is a decision better made by the consumer's
# policy than by this service.
CheckStatus = Literal["ok", "error"]


class ComponentHealth(BaseModel):
    """Health of one dependency.

    Attributes:
        status: ``"ok"`` if the probe succeeded, ``"error"`` otherwise.
        detail: Human-readable error message when ``status`` is
            ``"error"``. ``None`` on success. The detail is the raw
            exception string — useful for operators, potentially
            sensitive for external consumers. If this endpoint is
            exposed beyond the internal network, consider dropping the
            detail field or redacting it behind an auth gate.
    """

    status: CheckStatus
    detail: str | None = None


class HealthResponse(BaseModel):
    """Full health report for the service.

    Attributes:
        status: Aggregate status. ``"ok"`` only if every component is
            ``"ok"``; ``"error"`` if any component is ``"error"``.
        version: Application version, from ``settings.version``.
        app_name: Application name, from ``settings.app_name``.
        environment: Deployment environment (``"development"``,
            ``"production"``, etc.), from ``settings.environment``.
        components: Per-dependency health, keyed by component name
            (``"postgres"``, ``"redis"``). New dependencies are added
            to this dict without changing the response schema.
        uptime_seconds: Wall-clock seconds since the ASGI app started.
            Uses ``time.monotonic()``, so it's unaffected by system
            clock changes. Resets on every process restart — including
            a hot reload in development, where it will typically be
            very small.
    """

    status: CheckStatus
    version: str
    app_name: str
    environment: str
    components: dict[str, ComponentHealth]
    uptime_seconds: float = 0.0


async def _check_postgres(db: AsyncSession) -> ComponentHealth:
    """Probe Postgres with a trivial round-trip query.

    Runs ``select 1`` on the provided session. Any transport or SQL
    error counts as a failure — the endpoint doesn't distinguish
    "connection refused" from "auth failed" from "query timed out"
    because a load balancer only needs the binary answer.

    Args:
        db: The request's database session. Entered as an async
            context manager — see *Known limitations* below.

    Returns:
        ``ComponentHealth(status="ok")`` on success, or
        ``ComponentHealth(status="error", detail=...)`` with the raw
        exception message on failure.
    """
    try:
        async with db as conn:
            await conn.execute(text("select 1"))
        return ComponentHealth(status="ok")
    except (TimeoutError, PostgresError, SQLAlchemyError, OSError, ConnectionError) as e:
        log.warning("postgres health check failed", extra={"extra_fields": {"error": str(e)}})
        return ComponentHealth(status="error", detail=str(e))


async def _check_redis() -> ComponentHealth:
    """Probe Redis with a ``PING`` command.

    Args:
        (None — the Redis client is fetched from the module-level
        connection pool via ``get_redis()``.)

    Returns:
        ``ComponentHealth(status="ok")`` if PING returned, or
        ``ComponentHealth(status="error", detail=...)`` with the raw
        exception message on failure.

    Note:
        Only ``AuthenticationError`` and ``ConnectionRefusedError`` are
        caught. A timeout raises ``asyncio.TimeoutError``, which is a
        subclass of ``OSError`` in Python 3.11+ but not of either of
        these — it will propagate and produce a 500. If Redis becomes
        slow rather than unavailable, this endpoint returns 500
        instead of 503. See *Known limitations*.
    """
    try:
        await get_redis().ping()
        return ComponentHealth(status="ok")
    except (AuthenticationError, ConnectionRefusedError) as e:
        log.warning("redis health check failed", extra={"extra_fields": {"error": str(e)}})
        return ComponentHealth(status="error", detail=str(e))


@router.get("", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health(request: Request, response: Response, db: DbSession) -> HealthResponse:
    """Report the service's readiness to serve traffic.

    Probes Postgres and Redis in sequence and aggregates the results.
    The response body is always a ``HealthResponse``; the HTTP status
    is 200 if everything is healthy and 503 if any component is not.

    The ``response`` parameter is used to override the status code
    dynamically — FastAPI's ``status_code`` decorator argument is a
    static default, and this endpoint needs to return 200 *or* 503
    from the same handler.

    Args:
        request: Used to read ``app.state.start_time``, which the app
            sets on startup. If the app didn't set it, this raises
            ``AttributeError`` — the endpoint assumes a startup
            lifespan handler is in place.
        response: Mutated in place when a component fails, to set 503.
            FastAPI merges the returned model with this response
            object's status code.
        db: The request's database session, passed to
            ``_check_postgres``.

    Returns:
        A ``HealthResponse`` with per-component status, aggregate
        status, and uptime. The response body is identical for 200 and
        503; only the status code differs.
    """
    start = request.app.state.start_time
    components = {"postgres": await _check_postgres(db), "redis": await _check_redis()}

    overall: CheckStatus = "ok" if all(check.status == "ok" for check in components.values()) else "error"
    if overall == "error":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    uptime_seconds = round(time.monotonic() - start, 2)
    return HealthResponse(
        status=overall,
        environment=settings.environment,
        version=settings.version,
        app_name=settings.app_name,
        components=components,
        uptime_seconds=uptime_seconds,
    )
