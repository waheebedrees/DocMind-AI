from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.services.storage.base import (
    BaseStorage,
    InvalidKey,
    ObjectNotFound,
    StorageError,
    StoredObject,
    UnsupportedMime,
    UploadTooLarge,
)
from app.services.storage.local import LocalStorage

__all__ = [
    "BaseStorage",
    "StorageError",
    "StoredObject",
    "UnsupportedMime",
    "UploadTooLarge",
    "ObjectNotFound",
    "InvalidKey",
    "get_storage",
]


@lru_cache(maxsize=10)
def get_storage() -> BaseStorage:
    if settings.storage_backend == "local":
        return LocalStorage(Path(settings.storage_root))
    raise NotImplementedError(f"Storage backend '{settings.storage_backend}' is not implemented yet")
