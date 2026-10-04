"""Shared model cache for local sentence-transformers clients.

Both the embedder and the reranker load models into memory and need
to serialize inference calls. They share this cache to avoid two
independent registries with separate lifecycle management.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.core.logging import get_logger

log = get_logger(__name__)


ModelKind = Literal["embedder", "reranker"]


@dataclass(frozen=True, slots=True)
class ModelKey:
    """Identifies a distinct loaded model instance.

    `kind` is part of the key because two loader classes can load the
    same checkpoint for different purposes — a SentenceTransformer
    encodes text, a CrossEncoder scores query/document pairs. The
    resulting objects are not interchangeable.

    `max_length` only applies to CrossEncoder; use None for embedders.
    """

    kind: ModelKind
    model_name: str
    device: str
    max_length: int | None = None


_CACHE: dict[ModelKey, tuple[Any, threading.Lock]] = {}
_CACHE_LOCK = threading.Lock()


def get_or_load(
    key: ModelKey,
    loader: Callable[[], Any],
    *,
    log_context: dict[str, Any] | None = None,
) -> tuple[Any, threading.Lock]:
    """Return (model, encode_lock) for the given key, loading once.

    Double-checked locking: fast path avoids the module lock once the
    model is loaded; slow path ensures exactly one thread performs the
    load. Load failures are not cached, so a transient failure can be
    retried by the next caller.

    `loader` is called with no arguments and must raise on failure.
    Error classification is the loader's responsibility — this function
    only caches success and propagates failure untouched.
    """
    entry = _CACHE.get(key)
    if entry is not None:
        return entry

    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            return entry

        log.info("model_load_start", **(_log_fields(key, log_context)))
        model = loader()  # may raise; propagates without caching
        entry = (model, threading.Lock())
        _CACHE[key] = entry
        log.info("model_load_done", **(_log_fields(key, log_context)))
        return entry


def clear() -> None:
    """Drop all cached models. For tests or to reclaim memory."""
    with _CACHE_LOCK:
        _CACHE.clear()


def clear_kind(kind: ModelKind) -> None:
    """Drop cached models of a single kind. Useful in tests that only
    want to force one client to reload."""
    with _CACHE_LOCK:
        for key in [k for k in _CACHE if k.kind == kind]:
            del _CACHE[key]


def _log_fields(key: ModelKey, extra: dict[str, Any] | None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "kind": key.kind,
        "model": key.model_name,
        "device": key.device,
    }
    if key.max_length is not None:
        fields["max_length"] = key.max_length
    if extra:
        fields.update(extra)
    return fields
