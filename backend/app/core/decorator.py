import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from .exceptions import PermanentError
from .logging import get_logger

log = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def with_async_backoff(
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
    *,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Async retry with exponential backoff.

    PermanentError (and every subclass) is *never* retried, regardless
    of what `exceptions` says. That preserves the classification made
    by the wrapped function: a PermanentError means "this will not
    succeed on retry," and re-running it just wastes the retry budget
    and pollutes logs with failures that were already terminal.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for delay between retries.
        max_delay: Maximum delay in seconds.
        exceptions: Tuple of exceptions to catch and retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exc: BaseException | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except PermanentError:
                    log.warning(
                        "permanent failure in %s, not retrying",
                        func.__name__,
                    )
                    raise
                except exceptions as exc:
                    last_exc = exc
                    log.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt,
                        max_retries,
                        func.__name__,
                        exc,
                    )
                    if attempt == max_retries:
                        break
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
