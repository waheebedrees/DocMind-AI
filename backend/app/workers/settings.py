from datetime import datetime

from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage import build_storage
from app.workers.tasks import WORKER_FUNCTIONS

log = get_logger(__name__)


async def startup(ctx: dict):
    print("")
    ctx["started_at"] = datetime.now()
    ctx["storage"] = await build_storage().__aenter__()
    log.info("Worker starting up ...", env=settings.environment, backend=settings.storage_backend)


async def shutdown(ctx: dict):
    uptime = datetime.now() - ctx["started_at"]
    log.info(f"worker shutting down after {uptime} ")


async def scheduled_heartbeat(ctx) -> str:
    ts = datetime.now().isoformat(timespec="seconds")
    log.info(f"[HEARTBEAT] ({ts})")
    return ts


class WorkerSettings:
    redis_settings = RedisSettings(
        host=settings.redis_host or "redis",
        port=settings.redis_port,
        password=settings.redis_password,
    )
    max_jobs = 4
    job_timeout = 600  # 10 min hard ceiling per stage
    max_tries = 3
    retry_delay = 10  # seconds; ARQ applies exponential backoff
    keep_result = 3600  # 1 hour — Redis is transport, not truth
    retry_jobs = True
    health_check_interval = 30

    on_shutdown = shutdown
    on_startup = startup

    cron_jobs = [
        cron(scheduled_heartbeat, second={0, 30}),
    ]

    functions = WORKER_FUNCTIONS
