import asyncio
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from .logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


def with_async_backoff(
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
    *,
    max_delay: float = 60.6,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    async decorator for retrying functions with exponential backoff

    Args:
        max_retries (int): Maximum number of retry attempts
        initial_delay (float): Initial daily in seconds before first retry
        backoff_factor (float): Multiplier for delay between retries
        max_delay (float, optional): Maximum delay in seconds
        exceptions (tuple, optional): Tuple of exceptions to catch and retry

    Returns:
        Callable: Decorated  function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    log.warning("Attempt %d/%d failed for %s : %s", attempt, max_retries, func.__name__, exc)
                    if attempt == max_retries:
                        break
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_exc  # must re-raise, not return None

        return wrapper

    return decorator
