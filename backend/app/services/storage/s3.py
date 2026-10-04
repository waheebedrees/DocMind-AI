"""S3-compatible object storage backend."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client
    from types_aiobotocore_s3.type_defs import CompletedPartTypeDef


from app.core.logging import get_logger
from app.services.storage.base import (
    _CHUNK,
    _HEAD_BYTES,
    _KEY_RE,
    _PIPELINE_KEY_RE,
    BaseStorage,
    InvalidKey,
    ObjectNotFound,
    StorageError,
    StoredObject,
    UploadTooLarge,
)
from app.services.storage.mime import detect_mime, extension_for

from ._fn import error_code, safe_meta, unlink_quiet

log = get_logger(__name__)

# S3 multipart minimum (except the last part) is 5 MiB.
_PART_SIZE = 8 * 1024 * 1024


class S3Storage(BaseStorage):
    """Content-addressed storage on S3 (or MinIO / R2 / GCS-S3).

    Layout mirrors LocalStorage so keys are portable across backends:

        s3://{bucket}/{prefix}/{user_id}/{hash[:2]}/{hash}{ext}

    Dedup uses HeadObject + server-side copy is unnecessary: a PUT of
    identical bytes to the same key is a no-op for our purposes, and a
    HeadObject before upload avoids the transfer entirely.

    Does not subclass StorageBackend — same rationale as LocalStorage.
    """

    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str = "",
        chunk_size: int = _CHUNK,
        part_size: int = _PART_SIZE,
    ) -> None:
        if part_size < 5 * 1024 * 1024:
            raise ValueError("part_size must be >= 5 MiB (S3 multipart minimum)")
        self._client = client
        self._bucket = bucket
        # Normalize: no leading slash, exactly one trailing slash when non-empty.
        self._prefix = prefix.strip("/")
        if self._prefix:
            self._prefix += "/"
        self._chunk_size = chunk_size
        self._part_size = part_size

    async def startup(self) -> None:
        """Verify bucket access. Idempotent."""
        try:
            await self._client.head_bucket(Bucket=self._bucket)
        except Exception as e:
            raise StorageError(f"Cannot access bucket {self._bucket!r}: {e}") from e

    def key_for(self, *, user_id: UUID, content_hash: str, extension: str) -> str:
        # Logical key — the Protocol's key. Prefix is an implementation detail.
        return f"{user_id}/{content_hash[:2]}/{content_hash}{extension}"

    def _s3_key(self, key: str) -> str:
        """Logical key → physical S3 object key, after validation."""
        
        if not _KEY_RE.match(key):
            raise InvalidKey(f"Malformed key: {key!r}")
        return f"{self._prefix}{key}"

    async def put_stream(
        self,
        *,
        user_id: UUID,
        stream: AsyncIterator[bytes],
        filename: str,
        max_bytes: int,
    ) -> StoredObject:
        hasher = hashlib.sha256()
        size = 0
        head = bytearray()
        mime_type: str | None = None

        # Buffer the first part in memory so we can (a) sniff MIME early
        # and (b) decide single-shot PUT vs multipart without a temp file
        # for the common small-doc case.
        first_part = bytearray()
        upload_id: str | None = None
        parts: list[CompletedPartTypeDef] = []
        part_number = 0
        # Provisional S3 key unknown until we have the hash. Multipart
        # under a random staging key, then CopyObject into the final key.
        staging_key = f"{self._prefix}_multipart/{uuid4().hex}"

        async def flush_part(data: bytes, *, final: bool = False) -> None:
            nonlocal upload_id, part_number
            if not data and not final:
                return
            if upload_id is None:
                # Start multipart lazily — small files never get here.
                create_resp = await self._client.create_multipart_upload(
                    Bucket=self._bucket,
                    Key=staging_key,
                    ContentType=mime_type or "application/octet-stream",
                )
                upload_id = create_resp["UploadId"]
            part_number += 1
            part_resp = await self._client.upload_part(
                Bucket=self._bucket,
                Key=staging_key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=data,
            )
            parts.append({"ETag": part_resp["ETag"], "PartNumber": part_number})

        try:
            async for chunk in stream:
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge(f"Upload exceeds {max_bytes} bytes")
                hasher.update(chunk)

                if len(head) < _HEAD_BYTES:
                    head.extend(chunk[: _HEAD_BYTES - len(head)])
                    if len(head) >= _HEAD_BYTES:
                        mime_type = detect_mime(bytes(head), filename)

                first_part.extend(chunk)

                # Once past one part, spill to multipart.
                if upload_id is None and len(first_part) > self._part_size:
                    # Need MIME before create_multipart_upload tags it.
                    if mime_type is None:
                        mime_type = detect_mime(bytes(head), filename)
                    ready = bytes(first_part[: self._part_size])
                    del first_part[: self._part_size]
                    await flush_part(ready)
                elif upload_id is not None and len(first_part) >= self._part_size:
                    ready = bytes(first_part[: self._part_size])
                    del first_part[: self._part_size]
                    await flush_part(ready)

            if mime_type is None:
                mime_type = detect_mime(bytes(head), filename)

            content_hash = hasher.hexdigest()
            key = self.key_for(
                user_id=user_id,
                content_hash=content_hash,
                extension=extension_for(mime_type),
            )
            s3_key = self._s3_key(key)

            # Dedup: if this user already has the object, drop the upload.
            if await self._head_ok(s3_key):
                await self._abort_multipart(staging_key, upload_id)
                log.info("deduplicated_object", key=key, size_bytes=size)
                return StoredObject(
                    key=key,
                    filename=filename,
                    content_hash=content_hash,
                    mime_type=mime_type,
                    size_bytes=size,
                    deduplicated=True,
                )

            if upload_id is None:
                # Whole object fits in first_part — single PUT.
                await self._client.put_object(
                    Bucket=self._bucket,
                    Key=s3_key,
                    Body=bytes(first_part),
                    ContentType=mime_type,
                    Metadata={
                        "content-hash": content_hash,
                        "original-filename": safe_meta(filename),
                    },
                )
            else:
                # Final (possibly short) part, complete, then copy to
                # the content-addressed key and delete the staging object.
                if first_part:
                    await flush_part(bytes(first_part))
                await self._client.complete_multipart_upload(
                    Bucket=self._bucket,
                    Key=staging_key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
                upload_id = None  # completed — don't abort in finally
                await self._client.copy_object(
                    Bucket=self._bucket,
                    Key=s3_key,
                    CopySource={"Bucket": self._bucket, "Key": staging_key},
                    ContentType=mime_type,
                    MetadataDirective="REPLACE",
                    Metadata={
                        "content-hash": content_hash,
                        "original-filename": safe_meta(filename),
                    },
                )
                await self._client.delete_object(Bucket=self._bucket, Key=staging_key)

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
            await self._abort_multipart(staging_key, upload_id)
            raise

    async def _head_ok(self, s3_key: str) -> bool:
        try:
            await self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except self._client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        except Exception as e:
            # aiobotocore raises botocore.exceptions.ClientError; the
            # attribute path above works when the client is typed. Fall
            # back to a code sniff so plain moto/unittest mocks work.
            code = error_code(e)
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    async def _abort_multipart(self, staging_key: str, upload_id: str | None) -> None:
        if upload_id is None:
            return
        try:
            await self._client.abort_multipart_upload(
                Bucket=self._bucket,
                Key=staging_key,
                UploadId=upload_id,
            )
        # Cleanup path: the original error is already propagating. Catching
        # broadly so a failed abort (or a client that raises something outside
        # the botocore hierarchy, like the test double) never masks it.
        except Exception as e:  # noqa: BLE001
            log.warning(
                "multipart_abort_failed",
                key=staging_key,
                upload_id=upload_id,
                error=str(e),
            )

    def get(self, key: str) -> AsyncIterator[bytes]:
        s3_key = self._s3_key(key)  # eager key validation
        return self._iter_object(s3_key, key)

    async def open_stream(self, key: str) -> AsyncIterator[bytes]:
        s3_key = self._s3_key(key)
        if not await self._head_ok(s3_key):
            raise ObjectNotFound(f"Object not found: {key}")
        return self._iter_object(s3_key, key)

    async def _iter_object(self, s3_key: str, logical_key: str) -> AsyncIterator[bytes]:
        try:
            resp = await self._client.get_object(Bucket=self._bucket, Key=s3_key)
        except Exception as e:
            if error_code(e) in {"404", "NoSuchKey", "NotFound"}:
                raise ObjectNotFound(f"Object not found: {logical_key}") from e
            raise StorageError(f"S3 get_object failed: {e}") from e

        body = resp["Body"]
        try:
            while True:
                chunk = await body.read(self._chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    @asynccontextmanager
    async def materialize(self, key: str) -> AsyncIterator[Path]:
        """Download to a temp file; deleted on exit.

        Unlike LocalStorage this IS a copy. Callers still must not
        retain the path after the context exits.
        """
        s3_key = self._s3_key(key)
        fd, name = await asyncio.to_thread(
            tempfile.mkstemp,
            prefix="s3mat_",
            suffix=Path(key).suffix,
        )
        path = Path(name)
        try:
            # take ownership immediately; fd is invalid after this
            fh = os.fdopen(fd, "wb")
            try:
                try:
                    resp = await self._client.get_object(Bucket=self._bucket, Key=s3_key)
                except Exception as e:
                    if error_code(e) in {"404", "NoSuchKey", "NotFound"}:
                        raise ObjectNotFound(f"Object not found: {key}") from e
                    raise StorageError(f"S3 get_object failed: {e}") from e

                body = resp["Body"]
                try:
                    # write through the fd mkstemp already opened
                    while True:
                        chunk = await body.read(self._chunk_size)
                        if not chunk:
                            break
                        fh.write(chunk)
                finally:
                    body.close()
            finally:
                fh.close()

            yield path

        finally:
            await asyncio.to_thread(unlink_quiet, path)

    async def exists(self, key: str) -> bool:
        return await self._head_ok(self._s3_key(key))

    async def delete_raw(self, key: str) -> None:
        if not _PIPELINE_KEY_RE.match(key):
            raise InvalidKey(f"not a valid pipeline key: {key!r}")
        s3_key = f"{self._prefix}{key}" if self._prefix else key
        try:
            await self._client.delete_object(Bucket=self._bucket, Key=s3_key)
        except Exception as e:
            raise StorageError(f"S3 delete_object failed: {e}") from e

    async def list_keys(self) -> AsyncIterator[tuple[str, datetime]]:
        paginator = self._client.get_paginator("list_objects_v2")
        prefix = self._prefix or ""
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if prefix and key.startswith(prefix):
                    key = key[len(prefix) :]
                yield key, obj["LastModified"]

    async def delete(self, key: str) -> None:
        s3_key = self._s3_key(key)  # validates + applies prefix
        try:
            await self._client.delete_object(Bucket=self._bucket, Key=s3_key)
        except Exception as e:
            raise StorageError(f"S3 delete_object failed: {e}") from e
