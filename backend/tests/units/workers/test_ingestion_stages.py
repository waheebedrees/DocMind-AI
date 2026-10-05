"""Advanced unit tests for the ingestion stage implementations.

These tests focus on cross-cutting behavior rather than per-function
unit coverage:

- Artifact round-trips through the real gzip+JSON serializer, not the
  mocked helpers, so serialization regressions surface.
- Content-hash keying invariants: the same document content always
  produces the same storage keys, and different content never collides.
- Cache short-circuits: a cache hit must not re-run expensive work,
  but must still repair missing derived state (e.g. parsed_key).
- Failure isolation: a failure mid-stage must not leave half-written
  artifacts that a later stage would read as valid.
- Inter-stage contracts: extract writes what clean reads; chunk writes
  what embed reads; embed writes what index validates.
- Error classification: PermanentError vs TransientEmbeddingError vs
  PermanentEmbeddingError propagate unmodified, so run_stage routes
  them correctly.
- Idempotency: re-running a stage on the same document is safe.
"""

from __future__ import annotations

import gzip
import json
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import UUID, uuid4

import pytest
from docling_core.types.doc import DoclingDocument

from app.core.exceptions import (
    PermanentEmbeddingError,
    PermanentError,
    TransientEmbeddingError,
)
from app.db.repositories.chunks import PreparedChunk
from app.workers import tasks as T

DOC_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
OTHER_DOC_ID = UUID("ffffffff-1111-2222-3333-444444444444")
HASH_A = "a" * 64
HASH_B = "b" * 64


# --- builders -------------------------------------------------------


def chunk(
    i: int = 0,
    *,
    text: str | None = None,
    embedding: list[float] | None = None,
    dim: int = 384,
    token_count: int = 5,
    labels: tuple[str, ...] = ("p1",),
) -> PreparedChunk:
    return PreparedChunk(
        chunk_index=i,
        text=text if text is not None else f"chunk {i}",
        page_number=1,
        section="body",
        token_count=token_count,
        doc_item_labels=labels,
        embedding=embedding if embedding is not None else (
            [0.0] * dim if dim else None),
    )


def bare_chunk(i: int = 0, *, dim: int | None = None) -> PreparedChunk:
    """A chunk with no embedding — simulates a chunk-stage artifact."""
    return PreparedChunk(
        chunk_index=i,
        text=f"chunk {i}",
        page_number=None,
        section=None,
        token_count=5,
        doc_item_labels=(),
        embedding=None if dim is None else [0.0] * dim,
    )


def docling_doc(*, pages: int = 3, texts: int = 10, tables: int = 2) -> MagicMock:
    """A minimal DoclingDocument stand-in. Real serialization is stubbed
    on `model_dump_json`, but the counters used by `_docling_stats` are
    real lengths."""
    d = MagicMock()
    d.pages = [MagicMock() for _ in range(pages)]
    d.texts = [MagicMock() for _ in range(texts)]
    d.tables = [MagicMock() for _ in range(tables)]
    d.model_dump_json.return_value = json.dumps({"pages": pages})
    return d


def document(
    *,
    content_hash: str = HASH_A,
    storage_key: str = "users/u/x.pdf",
    parsed_key: str | None = "parsed/old",
) -> MagicMock:
    d = MagicMock()
    d.id = DOC_ID
    d.content_hash = content_hash
    d.storage_key = storage_key
    d.parsed_key = parsed_key
    return d


def materialize_cm(path: str = "/tmp/x.pdf"):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=path)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture
def storage() -> AsyncMock:
    s = AsyncMock()
    s.exists = AsyncMock(return_value=False)
    s.get_bytes = AsyncMock(return_value=b"")
    s.put_bytes = AsyncMock()
    # materialize() is a SYNC call that returns an async context manager
    # (via @asynccontextmanager). AsyncMock would make it return a
    # coroutine instead, which `async with` can't consume.
    s.materialize = MagicMock()
    return s

@pytest.fixture
def ctx(storage) -> dict:
    return {"storage": storage}


@pytest.fixture
def session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def ingester_patch():
    with patch.object(T, "Ingester") as cls:
        inst = AsyncMock()
        cls.return_value = inst
        yield inst


@pytest.fixture
def keys(monkeypatch):
    """Force deterministic key derivation so assertions don't depend on
    the real `app.services.storage.keys` layout."""
    kmap = {
        "parsed": lambda h: f"pipeline/{h}/parsed.json.gz",
        "clean": lambda h: f"pipeline/{h}/clean.json.gz",
        "chunks": lambda h: f"pipeline/{h}/chunks.json.gz",
        "embedded": lambda h: f"pipeline/{h}/embedded.json.gz",
    }
    monkeypatch.setattr(T, "parsed_key", kmap["parsed"])
    monkeypatch.setattr(T, "clean_key", kmap["clean"])
    monkeypatch.setattr(T, "chunks_key", kmap["chunks"])
    monkeypatch.setattr(T, "embedded_chunks_key", kmap["embedded"])
    return kmap


# ====================================================================
# Artifact round-trips: real serialization, not mocked helpers
# ====================================================================


class TestArtifactRoundTrip:
    async def test_parsed_document_survives_gzip_json_round_trip(self, storage):
        doc = docling_doc()
        await T.save_parsed_document(storage, doc, "k")
        written = storage.put_bytes.await_args.args[1]

        # The bytes on disk must be a valid gzip stream of the JSON
        decompressed = gzip.decompress(written)
        assert json.loads(decompressed) == {"pages": 3}

    async def test_load_parsed_document_reads_what_save_wrote(self, storage):
        """Round-trip both directions without mocking the model."""
        original = docling_doc()
        storage.put_bytes.side_effect = lambda _k, b: storage.get_bytes.__setattr__(
            "return_value", b
        )
        await T.save_parsed_document(storage, original, "k")

        with patch.object(
            DoclingDocument, "model_validate_json", return_value=original
        ) as mv:
            out = await T.load_parsed_document(storage, "k")

        assert out is original
        mv.assert_called_once()
        # Argument passed to model_validate_json must be the decompressed
        # JSON, not the raw gzip bytes.
        raw = mv.call_args.args[0]
        assert raw == b'{"pages": 3}'

    async def test_rows_survive_gzip_json_round_trip(self, storage):
        rows = [
            chunk(0, labels=("a", "b")),
            chunk(1, labels=()),
            chunk(2, labels=("single",)),
        ]
        await T._save_rows(storage, "k", rows)
        written = storage.put_bytes.await_args.args[1]

        # Simulate a storage backend: read back what was written
        storage.get_bytes.return_value = written
        loaded = await T._load_rows(storage, "k")

        assert len(loaded) == 3
        for original, restored in zip(rows, loaded):
            assert restored.chunk_index == original.chunk_index
            assert restored.text == original.text
            assert restored.token_count == original.token_count
            assert restored.doc_item_labels == original.doc_item_labels

    async def test_tuple_fields_become_tuples_not_lists_after_load(self, storage):
        """`asdict` emits tuples as JSON arrays; `_load_rows` must restore
        them so equality checks elsewhere don't break on list vs tuple."""
        rows = [chunk(0, labels=("x", "y"))]
        await T._save_rows(storage, "k", rows)
        storage.get_bytes.return_value = storage.put_bytes.await_args.args[1]

        loaded = await T._load_rows(storage, "k")
        assert isinstance(loaded[0].doc_item_labels, tuple)

    async def test_missing_doc_item_labels_defaults_to_empty_tuple(self, storage):
        payload = [
            {
                "chunk_index": 0,
                "text": "t",
                "page_number": None,
                "section": None,
                "token_count": 1,
                "embedding": None,
                # no doc_item_labels key
            }
        ]
        storage.get_bytes.return_value = gzip.compress(
            json.dumps(payload).encode())
        loaded = await T._load_rows(storage, "k")
        assert loaded[0].doc_item_labels == ()

    async def test_embeddings_survive_round_trip(self, storage):
        """Large float arrays are the heaviest part of the artifact; a
        precision or truncation bug here silently corrupts the index."""
        vec = [float(i) / 1000 for i in range(384)]
        rows = [chunk(0, embedding=vec)]
        await T._save_rows(storage, "k", rows)
        storage.get_bytes.return_value = storage.put_bytes.await_args.args[1]

        loaded = await T._load_rows(storage, "k")
        assert loaded[0].embedding == vec


# ====================================================================
# Content-hash keying: same hash → same keys, different hash → no collision
# ====================================================================


class TestContentHashKeying:
    async def test_same_content_hash_yields_same_parsed_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        storage.exists.return_value = False
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.extract_document = MagicMock(return_value=docling_doc())
        storage.materialize.return_value = materialize_cm()

        await T.do_extract(ctx, session, DOC_ID)

        expected = keys["parsed"](HASH_A)
        assert storage.put_bytes.await_args.args[0] == expected
        ingester_patch.set_parsed_key.assert_awaited_with(
            document_id=DOC_ID, parsed_key=expected
        )

    async def test_different_content_hash_yields_different_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_B)
        ingester_patch.extract_document = MagicMock(return_value=docling_doc())
        storage.materialize.return_value = materialize_cm()

        await T.do_extract(ctx, session, DOC_ID)

        assert storage.put_bytes.await_args.args[0] == keys["parsed"](HASH_B)

    async def test_clean_cache_hit_does_not_reparse(self, ctx, session, ingester_patch, storage, keys):
        """The whole point of content-addressed caching: if the clean
        artifact exists, don't re-read the parsed artifact, don't call
        the cleaner, don't write anything."""
        storage.exists.return_value = True
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)

        with (
            patch.object(T, "load_parsed_document", new=AsyncMock()) as load,
            patch.object(T, "save_parsed_document", new=AsyncMock()) as save,
        ):
            out = await T.do_clean(ctx, session, DOC_ID)

        assert out["cached"] is True
        load.assert_not_awaited()
        save.assert_not_awaited()
        ingester_patch.clean_document.assert_not_called()


# ====================================================================
# Cache-hit repairs: skip expensive work but fix missing derived state
# ====================================================================


class TestCacheHitRepairs:
    async def test_extract_cache_hit_recomputes_stats_from_artifact(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """On cache hit, stats must reflect the artifact, not be read
        from a sidecar. This is what makes the removal of `.meta.json`
        safe."""
        storage.exists.return_value = True
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        cached_doc = docling_doc(pages=7, texts=99, tables=4)

        with patch.object(T, "load_parsed_document", new=AsyncMock(return_value=cached_doc)):
            out = await T.do_extract(ctx, session, DOC_ID)

        assert out["pages"] == 7
        assert out["text_items"] == 99
        assert out["tables"] == 4

    async def test_extract_cache_hit_still_sets_parsed_key_on_document(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """Even on a hit, the document's parsed_key must be repaired if
        it's missing. Otherwise a crashed previous run leaves the
        downstream stage unable to find the artifact."""
        storage.exists.return_value = True
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A, parsed_key=None
        )

        with patch.object(T, "load_parsed_document", new=AsyncMock(return_value=docling_doc())):
            await T.do_extract(ctx, session, DOC_ID)

        ingester_patch.set_parsed_key.assert_awaited_once()

    async def test_extract_cache_miss_writes_then_sets_key_in_order(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """Order matters: write the artifact before recording its key.
        If we crash between the two, the document has no parsed_key and
        the next run will re-parse (idempotent). The reverse order would
        leave a dangling reference."""
        call_order = []
        storage.put_bytes.side_effect = lambda *_a, **_kw: call_order.append(
            "write")
        ingester_patch.set_parsed_key.side_effect = lambda **_kw: call_order.append(
            "set_key")

        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.extract_document = MagicMock(return_value=docling_doc())
        storage.materialize.return_value = materialize_cm()

        await T.do_extract(ctx, session, DOC_ID)

        assert call_order == ["write", "set_key"]


# ====================================================================
# Failure isolation: no half-written artifacts
# ====================================================================


class TestFailureIsolation:
    async def test_extract_parse_failure_writes_nothing(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """A Docling failure must not leave a partial artifact behind.
        The write happens *after* the parse, so this should hold."""
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.extract_document = MagicMock(
            side_effect=RuntimeError("PDF corrupt"))
        storage.materialize.return_value = materialize_cm()

        with pytest.raises(RuntimeError):
            await T.do_extract(ctx, session, DOC_ID)

        storage.put_bytes.assert_not_awaited()

    async def test_extract_write_failure_does_not_set_parsed_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """If storage write fails, parsed_key must not be set — otherwise
        downstream stages would try to read a nonexistent artifact."""
        storage.put_bytes.side_effect = OSError("disk full")
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.extract_document = MagicMock(return_value=docling_doc())
        storage.materialize.return_value = materialize_cm()

        with pytest.raises(OSError):
            await T.do_extract(ctx, session, DOC_ID)

        ingester_patch.set_parsed_key.assert_not_awaited()

    async def test_chunk_write_failure_leaves_no_chunk_artifact(
        self, ctx, session, ingester_patch, storage, keys
    ):
        storage.put_bytes.side_effect = OSError("disk full")
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.chunk_document = MagicMock(return_value=[chunk(0)])

        with patch.object(T, "load_parsed_document", new=AsyncMock(return_value=docling_doc())):
            with pytest.raises(OSError):
                await T.do_chunk(ctx, session, DOC_ID)

    async def test_index_validates_before_any_db_write(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """Validation must run before `bulk_insert`. If a row fails
        validation, no DELETE or INSERT should have been issued — the
        document keeps its previous (working) index."""
        rows = [
            chunk(0, embedding=[0.1] * 384),
            chunk(1, embedding=[0.1] * 128),   # wrong dim
            chunk(2, embedding=[0.1] * 384),
        ]
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)

        with (
            patch.object(T, "load_embedded_chunks",
                         new=AsyncMock(return_value=rows)),
            patch.object(T, "settings") as s,
        ):
            s.embedding_spec.dimension = 384
            with pytest.raises(PermanentError, match="dims"):
                await T.do_index(ctx, session, DOC_ID)

        ingester_patch.bulk_insert.assert_not_awaited()


# ====================================================================
# Error classification: run_stage routes on the exception type
# ====================================================================


class TestErrorClassification:
    async def test_embed_transient_error_propagates_unchanged(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """do_embed must not catch and re-wrap the exception — run_stage
        classifies based on the exact type."""
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.embed = AsyncMock(
            side_effect=TransientEmbeddingError("timeout")
        )

        with patch.object(T, "load_chunks", new=AsyncMock(return_value=[bare_chunk(0)])):
            with pytest.raises(TransientEmbeddingError):
                await T.do_embed(ctx, session, DOC_ID)

    async def test_embed_permanent_error_propagates_unchanged(
        self, ctx, session, ingester_patch, storage, keys
    ):
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.embed = AsyncMock(
            side_effect=PermanentEmbeddingError("dim mismatch")
        )

        with patch.object(T, "load_chunks", new=AsyncMock(return_value=[bare_chunk(0)])):
            with pytest.raises(PermanentEmbeddingError):
                await T.do_embed(ctx, session, DOC_ID)

    async def test_clean_missing_parsed_key_is_permanent(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """A missing parsed_key means extract never ran. That's a code
        path bug, not a transient failure — retrying won't help."""
        storage.exists.return_value = False
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A, parsed_key=None
        )

        with pytest.raises(PermanentError, match="parsed_key"):
            await T.do_clean(ctx, session, DOC_ID)


# ====================================================================
# Inter-stage contracts
# ====================================================================


class TestInterStageContracts:
    async def test_chunk_reads_clean_key_not_parsed_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """After clean runs, the chunk stage must read the *clean*
        artifact, not the raw parsed one."""
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.chunk_document = MagicMock(return_value=[chunk(0)])

        with patch.object(T, "load_parsed_document", new=AsyncMock()) as load:
            await T.do_chunk(ctx, session, DOC_ID)

        load.assert_awaited_once_with(storage, keys["clean"](HASH_A))

    async def test_embed_reads_chunks_key(self, ctx, session, ingester_patch, storage, keys):
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.embed = AsyncMock(return_value=[chunk(0, dim=384)])

        with patch.object(T, "load_chunks", new=AsyncMock(return_value=[])) as load:
            try:
                await T.do_embed(ctx, session, DOC_ID)
            except PermanentError:
                pass
            load.assert_awaited_once_with(storage, keys["chunks"](HASH_A))

    async def test_index_reads_embedded_chunks_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.bulk_insert = AsyncMock(return_value=(0, 0))

        with (
            patch.object(T, "load_embedded_chunks", new=AsyncMock(return_value=[])) as load,
            patch.object(T, "settings") as s,
        ):
            s.embedding_spec.dimension = 384
            try:
                await T.do_index(ctx, session, DOC_ID)
            except PermanentError:
                pass
            load.assert_awaited_once_with(storage, keys["embedded"](HASH_A))

    async def test_chunk_output_is_readable_by_embed(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """Round-trip through the actual serializer: what do_chunk writes
        must be what _load_rows reads back, field for field. This is the
        single most likely place a schema drift would go unnoticed."""
        original_chunks = [
            chunk(0, text="alpha", labels=("a", "b")),
            chunk(1, text="beta", labels=()),
            chunk(2, text="gamma", labels=("c",)),
        ]
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.chunk_document = MagicMock(return_value=original_chunks)

        captured = {}
        storage.put_bytes.side_effect = lambda k, b: captured.update({k: b})

        with patch.object(T, "load_parsed_document", new=AsyncMock(return_value=docling_doc())):
            await T.do_chunk(ctx, session, DOC_ID)

        storage.get_bytes.return_value = captured[keys["chunks"](HASH_A)]
        loaded = await T._load_rows(storage, keys["chunks"](HASH_A))

        assert len(loaded) == len(original_chunks)
        for o, l in zip(original_chunks, loaded):
            assert l.chunk_index == o.chunk_index
            assert l.text == o.text
            assert l.doc_item_labels == o.doc_item_labels


# ====================================================================
# Idempotency and re-execution safety
# ====================================================================


class TestIdempotency:
    async def test_extract_run_twice_yields_same_artifact_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.extract_document = MagicMock(return_value=docling_doc())
        storage.materialize.return_value = materialize_cm()

        await T.do_extract(ctx, session, DOC_ID)
        first_key = storage.put_bytes.await_args.args[0]

        storage.put_bytes.reset_mock()
        await T.do_extract(ctx, session, DOC_ID)
        second_key = storage.put_bytes.await_args.args[0]

        assert first_key == second_key

    async def test_index_replaces_previous_chunks(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """index is a full replacement, not an append. The result must
        report both the delete count and the insert count so the
        document's metadata can be reconciled."""
        rows = [chunk(0, dim=384), chunk(1, dim=384), chunk(2, dim=384)]
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.bulk_insert = AsyncMock(
            return_value=(5, 3))  # deleted 5, inserted 3

        with (
            patch.object(T, "load_embedded_chunks",
                         new=AsyncMock(return_value=rows)),
            patch.object(T, "settings") as s,
        ):
            s.embedding_spec.dimension = 384
            out = await T.do_index(ctx, session, DOC_ID)

        assert out == {"inserted": 3, "replaced": 5}

    async def test_clean_rerun_produces_same_key(
        self, ctx, session, ingester_patch, storage, keys
    ):
        storage.exists.return_value = False
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A, parsed_key="parsed/key"
        )
        ingester_patch.clean_document = MagicMock(return_value={"changed": 0})

        with patch.object(T, "load_parsed_document", new=AsyncMock(return_value=docling_doc())):
            await T.do_clean(ctx, session, DOC_ID)
            k1 = storage.put_bytes.await_args.args[0]

            storage.put_bytes.reset_mock()
            await T.do_clean(ctx, session, DOC_ID)
            k2 = storage.put_bytes.await_args.args[0]

        assert k1 == k2 == keys["clean"](HASH_A)


# ====================================================================
# Boundary conditions
# ====================================================================


class TestBoundaryConditions:
    async def test_docling_stats_for_empty_document(self):
        doc = docling_doc(pages=0, texts=0, tables=0)
        stats = T._docling_stats(doc)
        assert stats == {"pages": 0, "text_items": 0, "tables": 0}

    async def test_embed_empty_chunks_is_permanent(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """A document that produced zero chunks can't be embedded, and
        no amount of retrying will change that — must be Permanent."""
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)

        with patch.object(T, "load_chunks", new=AsyncMock(return_value=[])):
            with pytest.raises(PermanentError, match="no chunks"):
                await T.do_embed(ctx, session, DOC_ID)

    async def test_embed_zero_dimension_artifact_reported_as_zero(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """If a chunk somehow lacks an embedding, dimension should be
        reported as 0 rather than crashing. Index will reject it."""
        rows = [chunk(0, embedding=None, dim=0)]
        ingester_patch.get_document.return_value = document(
            content_hash=HASH_A)
        ingester_patch.embed = AsyncMock(return_value=rows)

        with (
            patch.object(T, "load_chunks", new=AsyncMock(
                return_value=[bare_chunk(0)])),
            patch.object(T, "_save_rows", new=AsyncMock()),
        ):
            out = await T.do_embed(ctx, session, DOC_ID)

        assert out["dimension"] == 0

    async def test_do_extract_uses_content_hash_not_document_id(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """Two documents with the same content must share the same
        artifact key. Otherwise dedup at upload time is wasted."""
        # First document extracts; second sees the cache.
        first = document(content_hash=HASH_A)
        second = document(content_hash=HASH_A)

        ingester_patch.get_document.return_value = first
        ingester_patch.extract_document = MagicMock(return_value=docling_doc())
        storage.materialize.return_value = materialize_cm()
        storage.exists.return_value = False
        await T.do_extract(ctx, session, DOC_ID)
        key1 = storage.put_bytes.await_args.args[0]

        ingester_patch.get_document.return_value = second
        storage.put_bytes.reset_mock()
        storage.exists.return_value = True
        with patch.object(T, "load_parsed_document", new=AsyncMock(return_value=docling_doc())):
            await T.do_extract(ctx, session, OTHER_DOC_ID)

        # Second run was a cache hit — no write, but same key advertised
        storage.put_bytes.assert_not_awaited()
        assert key1 == keys["parsed"](HASH_A)

    async def test_document_with_null_content_hash_still_keyable(
        self, ctx, session, ingester_patch, storage, keys
    ):
        """If content_hash is None the stage will crash on key
        derivation. That's a bug in upload — surface it, don't swallow."""
        ingester_patch.get_document.return_value = document(content_hash=None)
        with pytest.raises((TypeError, AttributeError)):
            await T.do_extract(ctx, session, DOC_ID)
