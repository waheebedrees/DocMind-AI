"""Local filesystem storage backend."""

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from app.core.logging import get_logger
from app.services.storage.base import (
    _CHUNK,
    _HEAD_BYTES,
    _KEY_RE,
    _PIPELINE_KEY_RE,
    _WRITE_BUFFER,
    validate_key,
    BaseStorage,
    InvalidKey,
    ObjectNotFound,
    StoredObject,
    UploadTooLarge,
)
from app.services.storage.mime import detect_mime, extension_for

from ._fn import flush_sync, rmdir_quiet, unlink_quiet

log = get_logger(__name__)


class LocalStorage(BaseStorage):
    """
    Content-addressed storage on the local filesystem.

    Layout:
        {root}/{user_id}/{hash[:2]}/{hash}{ext}   final objects
        {root}/tmp/{uuid}                          in-flight uploads

    Subclasses BaseStorage so the shared helpers (flush_sync,
    rmdir_quiet, unlink_quiet) are inherited. The Protocol's `...`
    methods are all overridden here, so the "silent None return"
    concern is moot.
    """

    def __init__(self, root: Path, chunk_size: int = _CHUNK) -> None:
        self._root = root
        self._root_resolved = root.resolve()
        self._tmp = root / "tmp"
        self._chunk_size = chunk_size

    async def startup(self) -> None:
        "create root and tmp directories. Idempotent"
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self._tmp.mkdir, parents=True, exist_ok=True)

    def key_for(self, *, user_id: UUID, content_hash: str, extension: str) -> str:
        return f"{user_id}/{content_hash[:2]}/{content_hash}{extension}"

    def _resolve(self, key: str) -> Path:
        """
        Map a key to a path . refusing anything outside the root

        `root / key` silently discards the root when the key is absolute/
        """
        validate_key(key)

        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root_resolved):
            raise InvalidKey(f" key escapes storage root: {key!r}")
        return path

    def get(self, key: str) -> AsyncIterator[bytes]:
        path = self._resolve(key)
        return self._iter_file(path, key)

    async def put_stream(
        self,
        *,
        user_id: UUID,
        stream: AsyncIterator[bytes],
        filename: str,
        max_bytes: int,
    ) -> StoredObject:

        tmp_path = self._tmp / uuid4().hex
        hasher = hashlib.sha256()
        size = 0
        head = bytearray()
        mime_type: str | None = None

        try:
            fh: BinaryIO = await asyncio.to_thread(tmp_path.open, "wb")
            try:
                buf = bytearray()
                async for chunk in stream:
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLarge(f"upload exceeds {max_bytes}")

                    hasher.update(chunk)
                    if len(head) < _HEAD_BYTES:
                        head.extend(chunk[: _HEAD_BYTES - len(head)])
                        if len(head) >= _HEAD_BYTES:
                            # Reject bad types before writing the
                            # remaining (potentially 50 MB) of body.
                            mime_type = detect_mime(bytes(head), filename)

                    buf.extend(chunk)
                    if len(buf) >= _WRITE_BUFFER:
                        await asyncio.to_thread(fh.write, bytes(buf))
                        buf.clear()
                if buf:
                    await asyncio.to_thread(fh.write, bytes(buf))
                    buf.clear()

                await asyncio.to_thread(flush_sync, fh)

            finally:
                await asyncio.to_thread(fh.close)

            # Files smaller than the head buffer never triggered the
            # early sniff.
            if mime_type is None:
                mime_type = detect_mime(bytes(head), filename)

            content_hash = hasher.hexdigest()
            key = self.key_for(user_id=user_id, content_hash=content_hash, extension=extension_for(mime_type))
            final_path = self._resolve(key)

            if await asyncio.to_thread(final_path.exists):
                await asyncio.to_thread(unlink_quiet, tmp_path)
                log.info("deduplicated_object", key=key, size_bytes=size)
                return StoredObject(
                    filename=filename,
                    key=key,
                    content_hash=content_hash,
                    mime_type=mime_type,
                    size_bytes=size,
                    deduplicated=True,
                )
            await asyncio.to_thread(
                final_path.parent.mkdir,
                parents=True,
                exist_ok=True,
            )
            await asyncio.to_thread(os.replace, tmp_path, final_path)

            log.info("stored_object", key=key, size_bytes=size, mime_type=mime_type)
            return StoredObject(
                key=key,
                filename=filename,
                content_hash=content_hash,
                mime_type=mime_type,
                size_bytes=size,
                deduplicated=False,
            )

        except BaseException:
            # any failure including client disconnect (Connection Error)
            # leave no partial state behind
            await asyncio.to_thread(unlink_quiet, tmp_path)
            raise

    async def _iter_file(self, path: Path, key: str) -> AsyncIterator[bytes]:
        if not await asyncio.to_thread(path.is_file):
            raise ObjectNotFound(f"object not found: {key}")

        fh: BinaryIO = await asyncio.to_thread(path.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(fh.read, self._chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(fh.close)

    async def open_stream(self, key: str) -> AsyncIterator[bytes]:
        path = self._resolve(key)
        if not await asyncio.to_thread(path.is_file):
            raise ObjectNotFound(f"object not found: {key}")

        return self._iter_file(path, key)

    @asynccontextmanager
    async def materialize(self, key: str) -> AsyncIterator[Path]:
        path = self._resolve(key)
        if not await asyncio.to_thread(path.is_file):
            raise ObjectNotFound(f"object not found: {key}")

        # No copy for LocalStorage — the real path. Read-only by
        # contract: writing through it corrupts content addressing.
        yield path


    async def exists(self, key: str) -> bool:
        """True if the object exists.

        Raises:
            InvalidKey: The key doesn't match the user or pipeline key shape,
                or it would escape the storage root.
        """
        path = self._resolve(key)
        return await asyncio.to_thread(path.is_file)


    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(unlink_quiet, path)
        # Let rmdir's own emptiness check race safely against a
        # concurrent put_stream; an explicit iterdir probe cannot.
        await asyncio.to_thread(rmdir_quiet, path.parent)


    async def put_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        await asyncio.to_thread(tmp.write_bytes, data)
        await asyncio.to_thread(os.replace, tmp, path)


    async def get_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        if not await asyncio.to_thread(path.is_file):
            raise ObjectNotFound(f"object not found: {key}")
        return await asyncio.to_thread(path.read_bytes)

    async def delete_raw(self, key: str) -> None:
        """Delete an internal pipeline artifact. Key must match _PIPELINE_KEY_RE."""
        if not _PIPELINE_KEY_RE.match(key):
            raise InvalidKey(f"not a valid pipeline key: {key!r}")
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root_resolved):
            raise InvalidKey(f"key escapes root: {key!r}")
        await asyncio.to_thread(unlink_quiet, path)

    async def list_keys(self) -> AsyncIterator[tuple[str, float]]:
        root = self._root_resolved
        tmp_dir = self._tmp.resolve()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.resolve().is_relative_to(tmp_dir):
                continue  # skip in-flight uploads

            yield path.relative_to(root).as_posix(), path.stat().st_mtime
