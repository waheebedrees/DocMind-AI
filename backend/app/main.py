from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.redis import close_redis
from app.db.session import engine

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("app_starting", env=settings.environment, app=settings.app_name, db_url=settings.database_url, redis_url=settings.redis_url)
    try:
        yield
    finally:
        log.info("app_shutdown")
        await engine.dispose()
        await close_redis()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
        debug=settings.debug,
        redoc_url=None,
        docs_url=settings.docs_url,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=settings.allow_methods,
        allow_headers=["Authorization", settings.api_key_headers],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["root"])
    async def root():
        return {
            "message": f"{settings.app_name} is running",
            "docs": "/docs",
            "version": settings.version,
            "health": "/api/v1/health",
        }

    return app


app = create_app()
