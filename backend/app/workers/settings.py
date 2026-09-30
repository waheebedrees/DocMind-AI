from arq.connections import RedisSettings

from app.core.config import settings


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 4
    job_timeout = 600  # 10 min hard ceiling per stage
    max_tries = 3
    retry_delay = 10  # seconds; ARQ applies exponential backoff
    keep_result = 3600  # 1 hour — Redis is transport, not truth
