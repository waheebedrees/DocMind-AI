"""Adapters for FastAPI's UploadFile."""

from collections.abc import AsyncIterator

from fastapi import UploadFile


async def iter_upload(upload: UploadFile, *, chunk_size: int) -> AsyncIterator[bytes]:
    """Yield the upload's bytes in fixed-size chunks.

    Does not seek or rewind — UploadFile is single-pass here.
    """
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        yield chunk
