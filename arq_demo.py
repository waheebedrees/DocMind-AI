
from arq import create_pool, cron, Retry
from arq.connections import RedisSettings
from arq.worker import JobExecutionFailed

import asyncio
import random
from datetime import datetime, timedelta
import sys 
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

load_dotenv("envs/dev.env")
import os 

os.environ['REDIS_HOST'] = 'localhost'


from backend.app.core.config import settings

MAX_TRIES = 3


# each job is an async function the first arg `ctx` shared context dit that persists jobs in the same worker 

async def say_hello(ctx:dict, name: str) -> str:
    
    msg = f"hello, {name}! (job #{ctx['job_try']})"
    print(msg)
    return msg

async def slow_task(ctx:dict, seconds: int = 3) -> str:
    print(f"staring slow_task {seconds} ...") 
    await asyncio.sleep(seconds)
    result = f"finishing after {seconds}"
    print(result)
    return result





async def flaky_task(ctx: dict, fail_rate: float = 0.7) -> str:
    attempt = ctx["job_try"]
    if random.random() < fail_rate:
        if attempt >= MAX_TRIES:
            raise RuntimeError(f"failed after {attempt} attempts")
        print(f"flaky_task failed on attempt {attempt}, retrying…")
        raise Retry(defer=0)  # or defer=attempt * 2 for backoff
    return f"succeeded on attempt {attempt}"

async def scheduled_heartbeat(ctx) -> str:
    ts = datetime.now().isoformat(timespec='seconds')
    print(f"[HEARTBEAT] ({ts})")
    return ts 





async def startup(ctx:dict):
    print("Worker starting up ...")
    ctx['started_at'] = datetime.now()
    
async def shutdown(ctx:dict):
    uptime = datetime.now() - ctx['started_at']
    print(f"worker shutting down after {uptime} ")

class WorkerSettings:
    functions = [say_hello, slow_task, flaky_task, scheduled_heartbeat]

    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
    )

    on_startup = startup
    on_shutdown = shutdown
    job_timeout = 30
    max_tries = MAX_TRIES  # 1 initial + 2 retries

    keep_result = 3600

    cron_jobs = [
        cron(scheduled_heartbeat, second={0, 30}),
    ]


async def producer():
    redis = await create_pool(WorkerSettings.redis_settings)
    
    print("\n--- Enqueuing jobs ---")

    j1 = await redis.enqueue_job('say_hello', 'waheeb')
    j2 = await redis.enqueue_job("slow_task", 2)
    j3 = await redis.enqueue_job("flaky_task", 0.7)
    
    

    print(f"Queued: {j1.job_id}, {j2.job_id}, {j3.job_id}")

    print("\n job status: (immediate)")
    for j in (j1, j2, j3):
        status = await j.status()
        
        print(f"{j._queue_name}    {j.job_id[:8]}... status={status}")


    print(f'\n waiting for result')
    try:
        r1 = await j1.result(timeout=10)
        print(f"  say_hello  -> {r1}")

    except BaseException as e:
        print(f" say_hello --> Error:{e}")

    try:
        r2 = await j2.result(timeout=10)
        print(f"  slow_task  -> {r2}")
    except BaseException as e:
        print(f"  slow_task  -> ERROR: {e}")

    try:
        r3 = await j3.result(timeout=15)
        print(f"  flaky_task -> {r3}")
    except JobExecutionFailed as e:
        print(f"  flaky_task -> gave up: {e}")
            
 # ── Enqueue a delayed job (run 5 seconds from now) ──
    run_at = datetime.now() + timedelta(seconds=5)
    j4 = await redis.enqueue_job("say_hello", "Later", _defer_until=run_at)
    print(
        f"\nDeferred job scheduled for {run_at.isoformat(timespec='seconds')}")

    await redis.aclose()

if __name__ == "__main__":
    asyncio.run(producer())
    
    
