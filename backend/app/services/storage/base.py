"""Storage backend protocol and shared types."""

import re
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

# {uuid}/{2 hex}/{64 hex}{.ext} — nothing else is a valid key.
_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"/[0-9a-f]{2}"
    r"/[0-9a-f]{64}\.[a-z0-9]{1,5}\Z"
)

_PIPELINE_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # user_id
    r"/_pipeline/"
    # document_id
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"/[a-z_]+\.json\.gz\Z"
)

_CHUNK = 64 * 1024
_WRITE_BUFFER = 4 * 1024 * 1024
_HEAD_BYTES = 8192


class StorageError(Exception):
    """Base class for storage failures."""


class ObjectNotFound(StorageError):
    """Raised when a key does not resolve to a stored object."""


class InvalidKey(StorageError):
    """Raised when a key is malformed or escapes the storage root."""


class UploadTooLarge(StorageError):
    """Raised mid-stream when the byte cap is exceeded."""


class UnsupportedMime(StorageError):
    """Raised when magic-byte detection yields a rejected MIME type."""


@dataclass(frozen=True)
class StoredObject:
    """Result of a successful put_stream call."""

    key: str
    content_hash: str
    mime_type: str
    size_bytes: int
    deduplicated: bool
    filename: str
    """True if this *user* already had an object with this hash.

    Dedup is per-user because user_id is part of the key. This is not
    a global content-existence signal; do not build billing on it.
    """


@runtime_checkable
class BaseStorage(Protocol):
    """Async object storage.

    Implementations must be safe for concurrent use from a single
    event loop.

    Keys are opaque to callers but MUST be validated by the
    implementation: they arrive from the database and, in a
    compromised or buggy caller, from user input. Implementations
    raise InvalidKey rather than touching anything outside their root.

    No method performs authorization. Verifying that the caller owns
    the user_id embedded in a key is the responsibility of the layer
    above.
    """

    async def put_stream(
        self,
        *,
        user_id: UUID,
        stream: AsyncIterator[bytes],
        filename: str,
        max_bytes: int,
    ) -> StoredObject:
        """Consume a byte stream, persist it, return metadata.

        The stream is consumed exactly once. Implementations enforce
        max_bytes during iteration, reject disallowed MIME types as
        early as the head buffer permits, and clean up partial state
        on any failure including cancellation.
        """
        ...

    async def open_stream(self, key: str) -> AsyncIterator[bytes]:
        """Validate key and existence eagerly, then return an iterator.

        Raises:
            InvalidKey, ObjectNotFound
        """
        ...

    def materialize(self, key: str) -> AbstractAsyncContextManager[Path]:
        """Yield a local filesystem path for the object.

        For local storage this is the real path (no copy) and MUST be
        treated as read-only. For remote storage this downloads to a
        temporary file deleted on exit. Callers must not retain the
        path after the context exits.
        """
        ...

    async def exists(self, key: str) -> bool: ...

    async def delete(self, key: str) -> None:
        """Remove the object. Idempotent: missing keys are not an error."""
        ...

    async def put_bytes(self, key: str, data: bytes) -> None:
        """Write raw bytes at an explicit key. for internal artifacts"""

    async def get_bytes(self, key: str) -> bytes:
        """Read the entire object as bytes. for internal artifacts"""

    async def delete_raw(self, key: str) -> None:
        """Trusted internal delete. Skips key validation. Idempotent.

        For pipeline artifacts written via write_sync — those keys don't
        match the user-key shape and would fail _KEY_RE enforcement.
        """
        ...

    def get(self, key: str) -> AsyncIterator[bytes]:
        """Return an async iterator over the object's bytes.

        Not a coroutine: call it, then `async for`. Key validation is
        eager (raises InvalidKey immediately); existence is only
        checked on first iteration. Use `open_stream` if you need a
        missing object to fail before response headers are sent.
        """
        ...

    def key_for(self, *, user_id: UUID, content_hash: str, extension: str) -> str:
        """Deterministic key derivation. Pure function."""
        ...

    def write_sync(self, key: str, data: bytes) -> None: ...

    def read_sync(self, key: str) -> bytes: ...
