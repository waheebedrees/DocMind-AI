"""Application entrypoint.

Builds and configures the FastAPI application:

- configures structured logging before any app object exists
- manages process-wide resources (storage backend, ARQ pool, DB engine,
  Redis client) through a single lifespan context
- wires CORS, the API middleware, and the versioned v1 router
- exposes a lightweight root endpoint for smoke checks

Importing this module does not open any network connections. All I/O
happens inside ``lifespan`` so that the module can be safely imported by
tests, CLI tools, and the ARQ worker without side effects.
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import APIMiddleware
from app.api.v1.router import api_router
from app.core.arq import create_arq_pool
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.redis import close_redis
from app.db.session import engine
from app.services.storage import build_storage

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage process-wide resources for the application's lifetime.

    Ordering matters and is deliberate:

    1. ``build_storage`` opens the object store (S3, local FS, etc.).
       It is an async context manager so the client is closed even if
       the ARQ pool or engine fail to initialise.
    2. ``start_time`` is recorded so ``/health`` and metrics middleware
       can report uptime without importing this module twice.
    3. ``create_arq_pool`` opens the Redis-backed ARQ pool used by the
       API to enqueue ingestion jobs. The pool is shared via
       ``app.state.arq_pool`` and must never be created per-request.
    4. The startup log line includes resolved URLs for observability;
       keep it last so it reflects a fully initialised app.

    Teardown runs in reverse dependency order: ARQ pool, then the SQL
    engine, then the Redis client. ``storage`` is closed automatically
    by its own context manager. Each ``close`` call is idempotent so a
    failed startup still produces a clean shutdown.
    """
    async with build_storage() as storage:
        app.state.start_time = time.monotonic()
        app.state.storage = storage

        arq_pool = await create_arq_pool()
        app.state.arq_pool = arq_pool

        log.info(
            "app_starting",
            env=settings.environment,
            app=settings.app_name,
            db_url=settings.database_url,
            redis_url=settings.redis_url,
        )
        try:
            yield
        finally:
            log.info("app_shutdown")
            await app.state.arq_pool.aclose()
            await engine.dispose()
            await close_redis()


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Exposed as a factory so tests can construct isolated app instances
    (with ``httpx.ASGITransport``) without sharing global state. The
    module-level ``app`` object is what Uvicorn/Gunicorn import.

    Middleware order is significant. Starlette applies middleware in
    reverse registration order for the request path, so the *last*
    ``add_middleware`` call is the *outermost* wrapper. Here:

    - ``APIMiddleware`` is registered last → runs first on requests,
      wraps logging, request-id propagation, and error envelope
      conversion around everything else.
    - ``CORSMiddleware`` runs inside it, so CORS responses still carry
      the request id and structured errors.

    Do not reorder these without understanding that inversion.
    """
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

    app.add_middleware(APIMiddleware)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        """Cheap unauthenticated probe.

        Intentionally not under ``/api/v1`` so load balancers and
        uptime checks do not depend on the versioned prefix. Does not
        touch the DB, Redis, or object storage — use
        ``/api/v1/health`` for a real dependency check.
        """
        return {
            "message": f"{settings.app_name} is running",
            "docs": "/docs",
            "version": settings.version,
            "health": "/api/v1/health",
        }

    return app


app = create_app()
