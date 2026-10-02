import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from app.core.config import StorageBackend, settings
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
from app.services.storage.s3 import S3Storage
from app.services.storage.uploads import iter_upload


@contextlib.asynccontextmanager
async def build_storage() -> AsyncIterator[BaseStorage]:
    """Construct the storage backend and yield it, releasing resources on exit."""
    if settings.storage_backend == StorageBackend.LOCAL:
        storage = LocalStorage(Path(settings.storage_root))
        await storage.startup()
        yield storage
        return


__all__ = [
    "BaseStorage",
    "StorageError",
    "StoredObject",
    "UnsupportedMime",
    "UploadTooLarge",
    "ObjectNotFound",
    "InvalidKey",
    "get_storage",
    "iter_upload",
    "S3Storage",
    "LocalStorage",
    "build_storage",
]
