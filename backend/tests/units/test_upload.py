import io

import pytest
from app.services.storage.uploads import iter_upload
from fastapi import UploadFile

pytestmark = pytest.mark.asyncio


async def test_iter_upload_chunks():
    up = UploadFile(filename="a.txt", file=io.BytesIO(b"abcdefghij"))
    chunks = [c async for c in iter_upload(up, chunk_size=4)]
    assert chunks == [b"abcd", b"efgh", b"ij"]


async def test_iter_upload_empty():
    up = UploadFile(filename="a.txt", file=io.BytesIO(b""))
    assert [c async for c in iter_upload(up, chunk_size=4)] == []
