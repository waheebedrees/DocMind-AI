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

router = APIRouter(tags=["health"])


CheckStatus = Literal["ok", "error"]


class ComponentHealth(BaseModel):
    status: CheckStatus
    detail: str | None = None


class HealthResponse(BaseModel):
    status: CheckStatus
    version: str
    app_name: str
    environment: str
    components: dict[str, ComponentHealth]
    uptime_seconds: float = 0.0


async def _check_postgres(db: AsyncSession) -> ComponentHealth:
    try:
        async with db as conn:
            await conn.execute(text("select 1"))
        return ComponentHealth(status="ok")
    except (TimeoutError, PostgresError, SQLAlchemyError, OSError, ConnectionError) as e:
        log.warning("postgres health check failed", extra={"extra_fields": {"error": str(e)}})
        return ComponentHealth(status="error", detail=str(e))


async def _check_redis() -> ComponentHealth:
    try:
        await get_redis().ping()
        return ComponentHealth(status="ok")
    except (AuthenticationError, ConnectionRefusedError) as e:
        log.warning("redis health check failed", extra={"extra_fields": {"error": str(e)}})
        return ComponentHealth(status="error", detail=str(e))


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health(request: Request, response: Response, db: DbSession) -> HealthResponse:
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
