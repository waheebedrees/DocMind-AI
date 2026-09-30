"""ARQ connection pool lifecycle."""

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import Request

from app.core.config import get_settings


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


async def create_arq_pool() -> ArqRedis:
    settings = get_settings()
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def close_arq_pool(pool: ArqRedis) -> None:
    await pool.aclose()
