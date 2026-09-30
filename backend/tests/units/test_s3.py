"""S3Storage tests against a fake client.

We deliberately do not pull moto into the unit suite: aiobotocore + moto
version skew is a recurring CI break, and the behaviour we care about
(key validation, multipart abort, dedup short-circuit, staging cleanup)
is all in our code, not in boto. A contract test against real MinIO lives
under tests/integration/ and is opt-in.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import UUID

import pytest
from app.services.storage.base import (
    BaseStorage,
    InvalidKey,
    ObjectNotFound,
    UnsupportedMime,
    UploadTooLarge,
)
from app.services.storage.s3 import S3Storage

pytestmark = pytest.mark.asyncio

USER = str(UUID("11111111-2222-3333-4444-555555555555"))
PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 1024


async def agen(*chunks: bytes) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


# --- minimal async S3 fake -------------------------------------------------


@dataclass
class FakeBody:
    data: bytes
    _pos: int = 0

    async def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self.data) - self._pos
        chunk = self.data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class FakeClientError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}


@dataclass
class S3Client:
    objects: dict[str, bytes] = field(default_factory=dict)
    content_types: dict[str, str] = field(default_factory=dict)
    multipart: dict[str, dict] = field(default_factory=dict)  # upload_id -> state
    aborted: list[str] = field(default_factory=list)
    _next_upload: int = 0

    @property
    def exceptions(self):
        return type("E", (), {"ClientError": FakeClientError})

    async def head_bucket(self, *, Bucket: str) -> dict:
        return {}

    async def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[Key])}

    async def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kw) -> dict:
        self.objects[Key] = bytes(Body)
        self.content_types[Key] = kw.get("ContentType", "")
        return {"ETag": '"abc"'}

    async def get_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": FakeBody(self.objects[Key])}

    async def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self.objects.pop(Key, None)
        return {}

    async def create_multipart_upload(self, *, Bucket: str, Key: str, **kw) -> dict:
        self._next_upload += 1
        uid = f"upload-{self._next_upload}"
        self.multipart[uid] = {"key": Key, "parts": {}, "content_type": kw.get("ContentType")}
        return {"UploadId": uid}

    async def upload_part(self, *, Bucket: str, Key: str, UploadId: str, PartNumber: int, Body: bytes) -> dict:
        self.multipart[UploadId]["parts"][PartNumber] = bytes(Body)
        return {"ETag": f'"part-{PartNumber}"'}

    async def complete_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str, MultipartUpload: dict) -> dict:
        state = self.multipart.pop(UploadId)
        parts = [state["parts"][p["PartNumber"]] for p in MultipartUpload["Parts"]]
        self.objects[Key] = b"".join(parts)
        self.content_types[Key] = state["content_type"] or ""
        return {}

    async def abort_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str) -> dict:
        self.multipart.pop(UploadId, None)
        self.aborted.append(UploadId)
        return {}

    async def copy_object(self, *, Bucket: str, Key: str, CopySource: dict, **kw) -> dict:
        src = CopySource["Key"]
        self.objects[Key] = self.objects[src]
        self.content_types[Key] = kw.get("ContentType", "")
        return {}


@pytest.fixture
def fake() -> S3Client:
    return S3Client()


@pytest.fixture
def storage(fake: S3Client) -> S3Storage:
    return S3Storage(fake, bucket="test-bucket", prefix="docmind", part_size=5 * 1024 * 1024)


# --- protocol --------------------------------------------------------------


async def test_satisfies_protocol(storage):
    assert isinstance(storage, BaseStorage)


@pytest.mark.parametrize(
    "key",
    [
        "/etc/passwd",
        "../x",
        "not-a-uuid/ab/" + "a" * 64 + ".txt",
        USER[:8] + "/ab/" + "a" * 64 + ".txt",  # type: ignore[index]
        f"{USER}/zz/" + "a" * 64 + ".txt",
        f"{USER}/ab/" + "a" * 63 + ".txt",
        f"{USER}/ab/" + "G" * 64 + ".txt",
        "",
    ],
)
async def test_bad_keys_rejected(storage, key):
    with pytest.raises(InvalidKey):
        await storage.exists(key)
    with pytest.raises(InvalidKey):
        await storage.delete(key)
    with pytest.raises(InvalidKey):
        storage.get(key)


async def test_prefix_applied(storage, fake):
    obj = await storage.put_stream(
        user_id=USER,
        stream=agen(b"hello world\n"),
        filename="a.txt",
        max_bytes=1_000,
    )
    assert obj.key.startswith(f"{USER}/")  # logical key has no prefix
    assert any(k.startswith("docmind/") for k in fake.objects)  # physical does
    assert f"docmind/{obj.key}" in fake.objects


# --- put / dedup / mime ----------------------------------------------------


async def test_put_small_single_shot(storage, fake):
    body = b"hello world\n"
    obj = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1_000)
    assert obj.deduplicated is False
    assert obj.content_hash == hashlib.sha256(body).hexdigest()
    assert fake.objects[f"docmind/{obj.key}"] == body
    assert fake.multipart == {}  # never started multipart


async def test_dedup_skips_upload(storage, fake):
    body = b"same bytes\n"
    first = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1_000)
    n_before = len(fake.objects)
    second = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1_000)
    assert second.deduplicated is True
    assert second.key == first.key
    assert len(fake.objects) == n_before


async def test_markdown_extension(storage, fake):
    obj = await storage.put_stream(
        user_id=USER,
        stream=agen(b"# Title\n\nbody\n"),
        filename="notes.md",
        max_bytes=1_000,
    )
    assert obj.mime_type == "text/markdown"
    assert obj.key.endswith(".md")


async def test_unsupported_mime_no_s3_write(storage, fake):
    elf = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 128
    with pytest.raises(UnsupportedMime):
        await storage.put_stream(user_id=USER, stream=agen(elf), filename="x.bin", max_bytes=1_000_000)
    assert fake.objects == {}
    assert fake.multipart == {}


async def test_too_large(storage, fake):
    with pytest.raises(UploadTooLarge):
        await storage.put_stream(
            user_id=USER,
            stream=agen(b"x" * 100),
            filename="a.txt",
            max_bytes=50,
        )
    assert fake.objects == {}


async def test_early_mime_reject_stops_stream(storage, fake):
    consumed = 0

    async def counting():
        nonlocal consumed
        yield b"\x7fELF" + b"\x00" * 9000
        consumed += 1
        yield b"z" * 1_000_000
        consumed += 1

    with pytest.raises(UnsupportedMime):
        await storage.put_stream(user_id=USER, stream=counting(), filename="x.bin", max_bytes=50_000_000)
    assert consumed == 0
    assert fake.objects == {}


# --- multipart path --------------------------------------------------------


async def test_multipart_for_large_payload(storage, fake):
    # part_size is 5 MiB; send 6 MiB so multipart engages.
    part = 5 * 1024 * 1024
    body_chunks = [b"a" * part, b"b" * (1 * 1024 * 1024)]
    total = b"".join(body_chunks)

    obj = await storage.put_stream(
        user_id=USER,
        stream=agen(*body_chunks),
        filename="big.txt",
        max_bytes=50 * 1024 * 1024,
    )
    assert fake.objects[f"docmind/{obj.key}"] == total
    assert fake.multipart == {}  # completed & popped
    # Staging key must not linger.
    assert not any("_multipart/" in k for k in fake.objects)


async def test_multipart_aborted_on_stream_error(storage, fake):
    part = 5 * 1024 * 1024

    async def boom():
        yield b"a" * (part + 1)  # forces multipart start
        raise RuntimeError("upstream died")

    with pytest.raises(RuntimeError):
        await storage.put_stream(user_id=USER, stream=boom(), filename="big.txt", max_bytes=50_000_000)
    assert fake.objects == {}
    assert fake.multipart == {}
    assert fake.aborted  # AbortMultipartUpload was called


async def test_multipart_aborted_on_dedup(storage, fake):
    """Large re-upload of an existing object must not leave staging debris."""
    part = 5 * 1024 * 1024
    body = b"x" * (part + 100)
    first = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=50_000_000)
    second = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=50_000_000)
    assert second.deduplicated is True
    assert second.key == first.key
    assert not any("_multipart/" in k for k in fake.objects)
    assert fake.multipart == {}


# --- reads / delete --------------------------------------------------------


async def test_get_roundtrip(storage):
    body = b"z" * 10_000
    obj = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1_000_000)
    got = b"".join([c async for c in storage.get(obj.key)])
    assert got == body


async def test_open_stream_missing(storage):
    h = "a" * 64
    with pytest.raises(ObjectNotFound):
        await storage.open_stream(f"{USER}/{h[:2]}/{h}.txt")


async def test_get_missing_raises_on_iterate(storage):
    h = "b" * 64
    it = storage.get(f"{USER}/{h[:2]}/{h}.txt")
    with pytest.raises(ObjectNotFound):
        await it.__anext__()


async def test_delete_idempotent(storage, fake):
    obj = await storage.put_stream(user_id=USER, stream=agen(b"bye\n"), filename="a.txt", max_bytes=100)
    await storage.delete(obj.key)
    await storage.delete(obj.key)
    assert not await storage.exists(obj.key)


async def test_materialize_downloads_and_cleans(storage, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    tempfile.tempdir = str(tmp_path)

    body = b"materialize on s3\n"
    obj = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1000)
    async with storage.materialize(obj.key) as path:
        assert path.read_bytes() == body
        assert path.exists()
        held = path
    assert not held.exists()
