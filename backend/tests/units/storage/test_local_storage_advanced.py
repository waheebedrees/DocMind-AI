"""Advanced tests for LocalStorage.

Uses the real filesystem (pytest's tmp_path), so path resolution,
symlinks, tmp cleanup, and shard pruning are exercised for real — not
against mocks. That is the point of the local backend: its correctness
lives in the filesystem semantics, not in our code alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.services.storage.base import (
    InvalidKey,
    ObjectNotFound,
    UnsupportedMime,
    UploadTooLarge,
)
from app.services.storage.local import LocalStorage

USER_ID = UUID("11111111-2222-3333-4444-555555555555")
OTHER_USER_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 1024
PDF = b"%PDF-1.4\n" + b"x" * 2048
ELF = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 128


async def agen(*chunks: bytes) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


def _valid_key(user_id: UUID, *, ext: str = ".txt", body: str | None = None) -> str:
    h = body if body is not None else ("a" * 64)
    return f"{user_id}/{h[:2]}/{h}{ext}"


@pytest.fixture
async def storage(tmp_path: Path) -> LocalStorage:
    s = LocalStorage(tmp_path)
    await s.startup()
    return s


# ====================================================================
# _resolve: path traversal and escape prevention
# ====================================================================


class TestResolve:
    def test_valid_user_key_resolves_under_root(self, storage):
        key = _valid_key(USER_ID)
        path = storage._resolve(key)
        assert path.is_relative_to(storage._root_resolved)
        assert path.name.endswith(".txt")

    def test_valid_pipeline_key_resolves(self, storage):
        key = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
        path = storage._resolve(key)
        assert path.is_relative_to(storage._root_resolved)

    def test_absolute_key_rejected(self, storage):
        with pytest.raises(InvalidKey):
            storage._resolve("/etc/passwd")

    def test_parent_traversal_rejected(self, storage):
        with pytest.raises(InvalidKey):
            storage._resolve("../../etc/passwd")

    def test_traversal_after_valid_prefix_rejected(self, storage):
        with pytest.raises(InvalidKey):
            storage._resolve(f"{USER_ID}/../etc/passwd")

    def test_empty_key_rejected(self, storage):
        with pytest.raises(InvalidKey):
            storage._resolve("")

    def test_random_string_rejected(self, storage):
        with pytest.raises(InvalidKey):
            storage._resolve("not-a-key")

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="symlink creation requires admin on Windows",
    )
    def test_symlink_escape_blocked(self, storage, tmp_path):
        """A symlink at a syntactically-valid key path must not let a
        read escape the root. This is the attack the `is_relative_to`
        check exists to stop; string-only validation would miss it."""
        h = "b" * 64
        subdir = storage._root / str(USER_ID) / h[:2]
        subdir.parent.mkdir(parents=True, exist_ok=True)
        outside = tmp_path.parent / "escape_target"
        outside.mkdir(exist_ok=True)
        try:
            subdir.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable in this environment")

        key = f"{USER_ID}/{h[:2]}/{h}.txt"
        with pytest.raises(InvalidKey, match="escapes"):
            storage._resolve(key)


# ====================================================================
# put_stream: lifecycle, dedup, cleanup
# ====================================================================


class TestPutStream:
    async def test_roundtrip(self, storage):
        body = b"hello world\n"
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        assert obj.content_hash == hashlib.sha256(body).hexdigest()
        assert obj.size_bytes == len(body)
        assert obj.deduplicated is False

        got = b"".join([c async for c in storage.get(obj.key)])
        assert got == body

    async def test_multichunk_stream_reassembled(self, storage):
        chunks = [b"a" * 100, b"b" * 100, b"c" * 100]
        body = b"".join(chunks)
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(*chunks), filename="a.txt", max_bytes=1_000_000)
        got = b"".join([c async for c in storage.get(obj.key)])
        assert got == body

    async def test_dedup_same_user(self, storage):
        body = b"same content\n"
        first = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        second = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        assert second.deduplicated is True
        assert second.key == first.key

    async def test_no_dedup_across_users(self, storage):
        body = b"same content\n"
        first = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        second = await storage.put_stream(user_id=OTHER_USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        assert second.deduplicated is False
        assert second.key != first.key

    async def test_markdown_gets_md_extension(self, storage):
        obj = await storage.put_stream(
            user_id=USER_ID,
            stream=agen(b"# Title\n\nbody\n"),
            filename="notes.md",
            max_bytes=1000,
        )
        assert obj.mime_type == "text/markdown"
        assert obj.key.endswith(".md")

    async def test_unsupported_mime_rejected_before_full_write(self, storage):
        """The early MIME sniff must fire before the entire payload is
        written. Otherwise a 500 MB binary wastes a full disk write."""
        consumed = 0

        async def counting():
            nonlocal consumed
            yield ELF + b"\x00" * 9000
            consumed += 1
            yield b"z" * 10_000_000
            consumed += 1

        with pytest.raises(UnsupportedMime):
            await storage.put_stream(
                user_id=USER_ID,
                stream=counting(),
                filename="x.bin",
                max_bytes=50_000_000,
            )
        assert consumed == 0
        # Nothing left behind
        for p in storage._tmp.iterdir():
            raise AssertionError(f"leaked tmp file: {p}")

    async def test_too_large_raises_and_cleans_tmp(self, storage):
        with pytest.raises(UploadTooLarge):
            await storage.put_stream(
                user_id=USER_ID,
                stream=agen(b"x" * 100),
                filename="a.txt",
                max_bytes=50,
            )
        assert list(storage._tmp.iterdir()) == []

    async def test_stream_error_cleans_tmp(self, storage):
        async def boom():
            yield b"a" * 100
            raise RuntimeError("upstream died")

        with pytest.raises(RuntimeError):
            await storage.put_stream(
                user_id=USER_ID,
                stream=boom(),
                filename="a.txt",
                max_bytes=1000,
            )
        assert list(storage._tmp.iterdir()) == []

    async def test_cancellation_cleans_tmp(self, storage):
        """A client disconnect mid-upload raises CancelledError, which
        is a BaseException. The `except BaseException` cleanup must
        still run."""
        started = asyncio.Event()

        async def slow():
            yield b"a" * 100
            started.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(
            storage.put_stream(
                user_id=USER_ID,
                stream=slow(),
                filename="a.txt",
                max_bytes=1000,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert list(storage._tmp.iterdir()) == []

    async def test_empty_upload_rejected(self, storage):
        """A zero-byte stream yields no head, so MIME detection fails."""
        with pytest.raises(UnsupportedMime):
            await storage.put_stream(
                user_id=USER_ID,
                stream=agen(),
                filename="a.txt",
                max_bytes=1000,
            )
        assert list(storage._tmp.iterdir()) == []

    async def test_large_write_crosses_buffer(self, storage):
        """Write buffer is 4 MiB; send 10 MiB to cross it at least twice."""
        body = os.urandom(10 * 1024 * 1024)
        # 10 MiB of random bytes isn't any allowed MIME — wrap in PDF header
        body = PDF + body
        obj = await storage.put_stream(
            user_id=USER_ID,
            stream=agen(body),
            filename="big.pdf",
            max_bytes=20 * 1024 * 1024,
        )
        got = b"".join([c async for c in storage.get(obj.key)])
        assert got == body


# ====================================================================
# get / open_stream / materialize
# ====================================================================


class TestReads:
    async def test_get_yields_all_bytes(self, storage):
        body = b"x" * 10_000
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1_000_000)
        got = b"".join([c async for c in storage.get(obj.key)])
        assert got == body

    async def test_get_respects_chunk_size(self, tmp_path):
        storage = LocalStorage(tmp_path, chunk_size=512)
        await storage.startup()
        body = b"y" * 4000
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1_000_000)
        sizes = [len(c) async for c in storage.get(obj.key)]
        assert sizes == [512, 512, 512, 512, 512, 512, 512, 416]

    async def test_get_rejects_bad_key_eagerly(self, storage):
        with pytest.raises(InvalidKey):
            storage.get("/etc/passwd")

    async def test_get_missing_raises_on_iteration(self, storage):
        key = _valid_key(USER_ID)
        it = storage.get(key)
        with pytest.raises(ObjectNotFound):
            await it.__anext__()

    async def test_open_stream_raises_eagerly_on_missing(self, storage):
        key = _valid_key(USER_ID)
        with pytest.raises(ObjectNotFound):
            await storage.open_stream(key)

    async def test_open_stream_roundtrip(self, storage):
        body = b"hello\n"
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        stream = await storage.open_stream(obj.key)
        got = b"".join([c async for c in stream])
        assert got == body

    async def test_materialize_yields_real_path(self, storage):
        body = b"materialize me\n"
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.txt", max_bytes=1000)
        async with storage.materialize(obj.key) as path:
            assert path.read_bytes() == body
            assert path.is_absolute()
        # Real path is not deleted after the context — unlike S3
        async with storage.materialize(obj.key) as path2:
            assert path2.read_bytes() == body

    async def test_materialize_missing_raises(self, storage):
        key = _valid_key(USER_ID)
        with pytest.raises(ObjectNotFound):
            async with storage.materialize(key):
                pass


# ====================================================================
# delete: idempotence and shard pruning
# ====================================================================


class TestDelete:
    async def test_delete_removes_object(self, storage):
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(PDF), filename="a.pdf", max_bytes=10_000)
        await storage.delete(obj.key)
        assert not await storage.exists(obj.key)

    async def test_delete_is_idempotent(self, storage):
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(PDF), filename="a.pdf", max_bytes=10_000)
        await storage.delete(obj.key)
        await storage.delete(obj.key)  # must not raise

    async def test_delete_removes_empty_shard(self, storage):
        """When the last object in a {hash[:2]} shard is deleted, the
        shard directory should be pruned. Otherwise content-addressed
        storage accumulates 256 empty dirs per user."""
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(PDF), filename="a.pdf", max_bytes=10_000)
        shard = storage._root / str(USER_ID) / obj.content_hash[:2]
        assert shard.is_dir()
        await storage.delete(obj.key)
        assert not shard.exists()

    async def test_delete_keeps_shard_with_siblings(self, storage):
        a = await storage.put_stream(user_id=USER_ID, stream=agen(PDF + b"a"), filename="a.pdf", max_bytes=10_000)
        b = await storage.put_stream(user_id=USER_ID, stream=agen(PDF + b"b"), filename="b.pdf", max_bytes=10_000)
        # Force the same shard by choosing a suffix that lands in the
        # same bucket — skip if not
        if a.content_hash[:2] != b.content_hash[:2]:
            pytest.skip("hashes landed in different shards")
        await storage.delete(a.key)
        assert await storage.exists(b.key)
        shard = storage._root / str(USER_ID) / a.content_hash[:2]
        assert shard.is_dir()

    async def test_delete_then_reupload(self, storage):
        body = PDF + b"content"
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.pdf", max_bytes=10_000)
        await storage.delete(obj.key)
        again = await storage.put_stream(user_id=USER_ID, stream=agen(body), filename="a.pdf", max_bytes=10_000)
        assert again.deduplicated is False
        assert again.key == obj.key


# ====================================================================
# put_bytes / get_bytes (internal artifacts)
# ====================================================================


class TestPutGetBytes:
    async def test_roundtrip(self, storage):
        key = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
        await storage.put_bytes(key, b"payload")
        assert await storage.get_bytes(key) == b"payload"

    async def test_get_bytes_missing_raises(self, storage):
        key = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
        with pytest.raises(ObjectNotFound):
            await storage.get_bytes(key)

    async def test_put_bytes_atomic_via_tmp_rename(self, storage):
        """The .tmp + os.replace pattern means a crash mid-write cannot
        leave a partial file at the final path. Verify the tmp file is
        gone and only the final exists."""
        key = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
        await storage.put_bytes(key, b"payload")
        final = storage._root / key
        tmp = final.with_suffix(final.suffix + ".tmp")
        assert final.exists()
        assert not tmp.exists()

    async def test_put_bytes_rejects_absolute_key(self, storage, tmp_path):
        """Path('/root') / '/tmp/x' returns '/tmp/x' on POSIX — the root
        is silently discarded. put_bytes must not allow this."""
        with pytest.raises(InvalidKey):
            await storage.put_bytes("/tmp/x", b"nope")

    async def test_put_bytes_rejects_traversal(self, storage):
        with pytest.raises(InvalidKey):
            await storage.put_bytes("../../escape", b"nope")

    async def test_get_bytes_rejects_absolute_key(self, storage):
        with pytest.raises(InvalidKey):
            await storage.get_bytes("/etc/passwd")


# ====================================================================
# delete_raw: pipeline keys only
# ====================================================================


class TestDeleteRaw:
    async def test_deletes_pipeline_key(self, storage):
        key = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
        await storage.put_bytes(key, b"x")
        await storage.delete_raw(key)
        assert not (storage._root / key).exists()

    async def test_delete_raw_is_idempotent(self, storage):
        key = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
        await storage.delete_raw(key)  # missing — must not raise
        await storage.delete_raw(key)

    async def test_rejects_user_key(self, storage):
        with pytest.raises(InvalidKey):
            await storage.delete_raw(_valid_key(USER_ID))

    async def test_rejects_absolute_key(self, storage):
        with pytest.raises(InvalidKey):
            await storage.delete_raw("/etc/passwd")


# ====================================================================
# list_keys
# ====================================================================


class TestListKeys:
    async def test_yields_all_user_objects(self, storage):
        a = await storage.put_stream(user_id=USER_ID, stream=agen(PDF + b"a"), filename="a.pdf", max_bytes=10_000)
        b = await storage.put_stream(user_id=USER_ID, stream=agen(PDF + b"b"), filename="b.pdf", max_bytes=10_000)
        keys = [k async for k, _ in storage.list_keys()]
        assert a.key in keys
        assert b.key in keys

    async def test_skips_tmp(self, storage):
        await storage.put_stream(user_id=USER_ID, stream=agen(PDF), filename="a.pdf", max_bytes=10_000)
        # Simulate an in-flight upload
        in_flight = storage._tmp / uuid4().hex
        in_flight.write_bytes(b"in progress")
        keys = [k async for k, _ in storage.list_keys()]
        assert all("tmp" not in k for k in keys)

    async def test_yields_mtime(self, storage):
        await storage.put_stream(user_id=USER_ID, stream=agen(PDF), filename="a.pdf", max_bytes=10_000)
        items = [item async for item in storage.list_keys()]
        assert len(items) >= 1
        _, mtime = items[0]
        assert isinstance(mtime, float)
        assert mtime > 0


# ====================================================================
# exists
# ====================================================================


class TestExists:
    async def test_true_for_present_object(self, storage):
        obj = await storage.put_stream(user_id=USER_ID, stream=agen(PDF), filename="a.pdf", max_bytes=10_000)
        assert await storage.exists(obj.key) is True

    async def test_false_for_missing_object(self, storage):
        assert await storage.exists(_valid_key(USER_ID)) is False

    async def test_exists_raises_for_invalid_key(self, storage):
        """exists() validates the key shape and raises rather than
        returning False — same contract as get() and delete()."""
        with pytest.raises(InvalidKey):
            await storage.exists("/etc/passwd")


# ====================================================================
# startup
# ====================================================================


class TestStartup:
    async def test_startup_creates_root_and_tmp(self, tmp_path):
        s = LocalStorage(tmp_path / "brand-new")
        assert not (tmp_path / "brand-new").exists()
        await s.startup()
        assert (tmp_path / "brand-new").is_dir()
        assert (tmp_path / "brand-new" / "tmp").is_dir()

    async def test_startup_is_idempotent(self, tmp_path):
        s = LocalStorage(tmp_path)
        await s.startup()
        await s.startup()
        assert (tmp_path / "tmp").is_dir()
