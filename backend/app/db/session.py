from collections.abc import AsyncGenerator

from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def resolve_async_url(url: str) -> str:
    parsed = make_url(url)
    driver_map = {
        "postgresql": "postgresql+asyncpg",
        "postgres": "postgresql+asyncpg",
        "postgresql+psycopg2": "postgresql+asyncpg",
    }
    driver = parsed.drivername
    if driver in driver_map:
        return str(parsed.set(drivername=driver_map[driver]))
    if "+asyncpg" in driver:
        return url
    raise ValueError(f"Unsupported database driver: {driver}")


engine: AsyncEngine = create_async_engine(
    resolve_async_url(settings.database_url),
    echo=settings.db_echo,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            if session.in_transaction():
                await session.rollback()
            raise
