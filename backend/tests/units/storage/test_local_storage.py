import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.services.storage.base import (
    BaseStorage,
    InvalidKey,
    ObjectNotFound,
    UnsupportedMime,
    UploadTooLarge,
)
from app.services.storage.local import LocalStorage

pytestmark = pytest.mark.asyncio

USER = UUID("11111111-2222-3333-4444-555555555555")
OTHER_USER = UUID("99999999-8888-7777-6666-555555555555")
PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 1024


async def agen(*chunks: bytes) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


def tmp_files(storage: LocalStorage) -> list[Path]:
    return list((storage._tmp).iterdir())


@pytest.fixture
async def storage(tmp_path: Path) -> LocalStorage:
    s = LocalStorage(tmp_path / "objects")
    await s.startup()
    return s


async def store_text(storage, text: bytes = b"hello world\n", name="a.txt"):
    return await storage.put_stream(user_id=USER, stream=agen(text), filename=name, max_bytes=1_000_000)


async def test_satisfies_protocol(storage):
    assert isinstance(storage, BaseStorage)


TRAVERSAL_KEYS = [
    "/etc/passwd",
    "../../etc/passwd",
    f"{USER}/../../../etc/passwd",
    f"{USER}/ab/../../../../etc/passwd",
    "",
    "not-a-key",
    f"{USER}/ab/{'z' * 64}.txt",  # non-hex hash
    f"{USER}/abc/{'a' * 64}.txt",  # wrong shard width
    f"{USER}/ab/{'a' * 63}.txt",  # short hash
    "ZZZZZZZZ-2222-3333-4444-555555555555/ab/" + "a" * 64 + ".txt",
]


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
async def test_exists_rejects_bad_keys(storage, key):
    with pytest.raises(InvalidKey):
        await storage.exists(key)


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
async def test_delete_rejects_bad_keys(storage, key):
    with pytest.raises(InvalidKey):
        await storage.delete(key)


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
async def test_get_rejects_bad_keys_eagerly(storage, key):
    # Must raise on call, not on first __anext__.
    with pytest.raises(InvalidKey):
        storage.get(key)


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
async def test_materialize_rejects_bad_keys(storage, key):
    with pytest.raises(InvalidKey):
        async with storage.materialize(key):
            pass


async def test_traversal_cannot_delete_outside_root(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("do not delete me")
    storage = LocalStorage(tmp_path / "objects")
    await storage.startup()

    with pytest.raises(InvalidKey):
        await storage.delete("../victim.txt")
    assert victim.exists()


async def test_traversal_cannot_read_outside_root(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")
    storage = LocalStorage(tmp_path / "objects")
    await storage.startup()

    with pytest.raises(InvalidKey):
        storage.get(str(secret))


async def test_symlink_escape_blocked(tmp_path):
    """A well-formed key whose resolved path leaves the root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("leaked")

    storage = LocalStorage(tmp_path / "objects")
    await storage.startup()

    h = "a" * 64
    shard = storage._root / str(USER) / h[:2]
    shard.mkdir(parents=True)
    try:
        (shard / f"{h}.txt").symlink_to(outside / "leak.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(InvalidKey):
        await storage.exists(f"{USER}/{h[:2]}/{h}.txt")


# --- put_stream happy paths ----------------------------------------------


async def test_put_stream_stores_and_hashes(storage):
    body = b"hello world\n"
    obj = await store_text(storage, body)

    assert obj.content_hash == hashlib.sha256(body).hexdigest()
    assert obj.size_bytes == len(body)
    assert obj.mime_type == "text/plain"
    assert obj.deduplicated is False
    assert obj.key == f"{USER}/{obj.content_hash[:2]}/{obj.content_hash}.txt"
    assert await storage.exists(obj.key)
    assert (storage._root / obj.key).read_bytes() == body
    assert tmp_files(storage) == []


async def test_multi_chunk_stream_reassembled(storage):
    chunks = [b"abc", b"def", b"ghi"]
    obj = await storage.put_stream(user_id=USER, stream=agen(*chunks), filename="a.txt", max_bytes=1_000_000)
    assert (storage._root / obj.key).read_bytes() == b"abcdefghi"
    assert obj.size_bytes == 9


async def test_large_multi_buffer_write(storage):
    """Exercise the 4 MB flush path."""
    chunk = b"x" * (64 * 1024)
    n = 160  # 10 MB
    obj = await storage.put_stream(
        user_id=USER,
        stream=agen(*([chunk] * n)),
        filename="big.txt",
        max_bytes=50 * 1024 * 1024,
    )
    assert obj.size_bytes == len(chunk) * n
    assert (storage._root / obj.key).stat().st_size == len(chunk) * n


async def test_markdown_gets_md_extension(storage):
    obj = await storage.put_stream(
        user_id=USER,
        stream=agen(b"# Title\n\nbody\n"),
        filename="notes.md",
        max_bytes=1_000_000,
    )
    assert obj.mime_type == "text/markdown"
    assert obj.key.endswith(".md")


async def test_dedup_same_user(storage):
    first = await store_text(storage)
    second = await store_text(storage)

    assert second.deduplicated is True
    assert second.key == first.key
    assert tmp_files(storage) == []


async def test_no_dedup_across_users(storage):
    body = b"shared content\n"
    a = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1_000)
    b = await storage.put_stream(user_id=OTHER_USER, stream=agen(body), filename="a.txt", max_bytes=1_000)
    assert a.content_hash == b.content_hash
    assert a.key != b.key
    assert b.deduplicated is False


# --- put_stream failure paths --------------------------------------------


async def test_too_large_raises_and_cleans_tmp(storage):
    with pytest.raises(UploadTooLarge):
        await storage.put_stream(
            user_id=USER,
            stream=agen(b"x" * 10, b"y" * 10),
            filename="a.txt",
            max_bytes=15,
        )
    assert tmp_files(storage) == []


async def test_unsupported_mime_cleans_tmp(storage):
    elf = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 128
    with pytest.raises(UnsupportedMime):
        await storage.put_stream(user_id=USER, stream=agen(elf), filename="mal.bin", max_bytes=1_000_000)
    assert tmp_files(storage) == []


async def test_empty_upload_rejected(storage):
    with pytest.raises(UnsupportedMime, match="Empty file"):
        await storage.put_stream(user_id=USER, stream=agen(), filename="empty.txt", max_bytes=1_000)
    assert tmp_files(storage) == []


async def test_mime_rejected_before_full_body_written(storage):
    """The early sniff must fire at the head boundary, not at EOF."""
    consumed = 0

    async def counting_stream():
        nonlocal consumed
        yield b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 9000
        consumed += 1
        yield b"z" * 1_000_000
        consumed += 1

    with pytest.raises(UnsupportedMime):
        await storage.put_stream(
            user_id=USER,
            stream=counting_stream(),
            filename="mal.bin",
            max_bytes=50 * 1024 * 1024,
        )
    assert consumed == 0  # never pulled the second chunk
    assert tmp_files(storage) == []


async def test_cancellation_cleans_tmp(storage):
    async def dying_stream():
        yield b"partial data here\n"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await storage.put_stream(
            user_id=USER,
            stream=dying_stream(),
            filename="a.txt",
            max_bytes=1_000_000,
        )
    assert tmp_files(storage) == []


async def test_stream_error_cleans_tmp(storage):
    async def broken_stream():
        yield b"some data\n"
        raise RuntimeError("upstream exploded")

    with pytest.raises(RuntimeError):
        await storage.put_stream(
            user_id=USER,
            stream=broken_stream(),
            filename="a.txt",
            max_bytes=1_000_000,
        )
    assert tmp_files(storage) == []


# --- get / open_stream ----------------------------------------------------


async def test_get_yields_all_bytes(storage):
    body = b"z" * (200 * 1024)
    obj = await storage.put_stream(user_id=USER, stream=agen(body), filename="a.txt", max_bytes=1_000_000)
    out = b"".join([c async for c in storage.get(obj.key)])
    assert out == body


async def test_get_respects_chunk_size(tmp_path):
    s = LocalStorage(tmp_path / "objects", chunk_size=1024)
    await s.startup()
    obj = await s.put_stream(user_id=USER, stream=agen(b"q" * 5000), filename="a.txt", max_bytes=1_000_000)
    sizes = [len(c) async for c in s.get(obj.key)]
    assert sizes == [1024, 1024, 1024, 1024, 904]


async def test_get_missing_raises_on_iteration(storage):
    h = "b" * 64
    key = f"{USER}/{h[:2]}/{h}.txt"
    it = storage.get(key)  # valid shape, no file
    with pytest.raises(ObjectNotFound):
        await it.__anext__()


async def test_open_stream_raises_eagerly(storage):
    h = "c" * 64
    key = f"{USER}/{h[:2]}/{h}.txt"
    with pytest.raises(ObjectNotFound):
        await storage.open_stream(key)


async def test_open_stream_reads(storage):
    obj = await store_text(storage, b"streamed\n")
    it = await storage.open_stream(obj.key)
    assert b"".join([c async for c in it]) == b"streamed\n"


# --- materialize ----------------------------------------------------------


async def test_materialize_yields_real_path(storage):
    body = b"materialize me\n"
    obj = await store_text(storage, body)
    async with storage.materialize(obj.key) as p:
        assert p.is_file()
        assert p.read_bytes() == body
        assert p == storage._root / obj.key
    # LocalStorage does not copy, so the file survives the context.
    assert (storage._root / obj.key).exists()


async def test_materialize_missing_raises(storage):
    h = "d" * 64
    with pytest.raises(ObjectNotFound):
        async with storage.materialize(f"{USER}/{h[:2]}/{h}.txt"):
            pass


# --- delete ---------------------------------------------------------------


async def test_delete_removes_object_and_empty_shard(storage):
    obj = await store_text(storage)
    shard = (storage._root / obj.key).parent

    await storage.delete(obj.key)

    assert not await storage.exists(obj.key)
    assert not shard.exists()


async def test_delete_keeps_shard_with_siblings(storage):
    obj = await store_text(storage)
    shard = (storage._root / obj.key).parent
    sibling = shard / ("f" * 64 + ".txt")
    sibling.write_bytes(b"sibling")

    await storage.delete(obj.key)

    assert shard.exists()
    assert sibling.exists()


async def test_delete_is_idempotent(storage):
    """Regression: Path.rmdir() takes no missing_ok kwarg."""
    obj = await store_text(storage)
    await storage.delete(obj.key)
    await storage.delete(obj.key)  # must not raise TypeError or OSError
    assert not await storage.exists(obj.key)


async def test_delete_then_reupload(storage):
    obj = await store_text(storage)
    await storage.delete(obj.key)
    again = await store_text(storage)
    assert again.deduplicated is False
    assert again.key == obj.key


# --- misc -----------------------------------------------------------------


async def test_key_for_is_pure():
    s = LocalStorage(Path("/nonexistent"))
    h = "a" * 64
    assert s.key_for(user_id=USER, content_hash=h, extension=".pdf") == (f"{USER}/aa/{h}.pdf")


async def test_generated_keys_always_validate(storage):
    """key_for output must survive _resolve for every extension."""
    from app.services.storage.mime import _MIME_TO_EXT

    h = "a" * 64
    for ext in {*_MIME_TO_EXT.values(), ".bin"}:
        key = storage.key_for(user_id=uuid4(), content_hash=h, extension=ext)
        storage._resolve(key)  # must not raise


async def test_startup_is_idempotent(storage):
    await storage.startup()
    await storage.startup()
    assert storage._tmp.is_dir()


