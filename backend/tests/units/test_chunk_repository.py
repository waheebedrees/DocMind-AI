from app.models.chunk import DocumentChunk
from app.db.repositories.chunks import ChunkRepository, ChunkRow
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import CompileError, IntegrityError
from sqlalchemy import select
import pytest
import uuid
from app.core.config import settings
from app.db.repositories.chunks import ChunkRow

from tests.units.conftest import  make_document, make_user

def _vec(seed: float = 0.0) -> list[float]:
    return [seed] * settings.embedding_dim


def make_chunk_row(
    *,
    chunk_index: int = 0,
    text: str = "hello world",
    page_number: int | None = 1,
    section: str | None = None,
    token_count: int = 10,
    doc_item_labels: tuple[str, ...] = (),
    embedding: list[float] | None = None,
) -> ChunkRow:
    return ChunkRow(
        chunk_index=chunk_index,
        text=text,
        page_number=page_number,
        section=section,
        token_count=token_count,
        doc_item_labels=doc_item_labels,
        embedding=embedding if embedding is not None else _vec(),
    )


pytestmark = pytest.mark.unit


async def _persist(db: AsyncSession, *objs):
    for o in objs:
        db.add(o)
    await db.flush()
    for o in objs:
        await db.refresh(o)
    return objs if len(objs) > 1 else objs[0]


async def _a_document(db: AsyncSession):
    user = await _persist(db, make_user())
    return await _persist(db, make_document(user.id))


# ─────────────────────────────────────────────────────────────
# bulk_insert
# ─────────────────────────────────────────────────────────────

class TestBulkInsert:
    async def test_inserts_all_rows_and_returns_count(self, db: AsyncSession):
        """Catches the doc_item_labels CompileError bug."""
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        rows = [
            make_chunk_row(chunk_index=i, text=f"chunk {i}")
            for i in range(5)
        ]

        count = await repo.bulk_insert(doc.id, rows)

        assert count == 5
        assert await repo.count_for_document(doc.id) == 5

    async def test_empty_rows_returns_zero_and_does_not_touch_db(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)

        count = await repo.bulk_insert(doc.id, [])

        assert count == 0
        assert await repo.count_for_document(doc.id) == 0

    async def test_rows_are_retrievable_with_all_fields(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        embedding = [0.01] * settings.embedding_dim
        row = make_chunk_row(
            chunk_index=7,
            text="specific text",
            page_number=3,
            section="intro",
            token_count=42,
            doc_item_labels=("a", "b"),
            embedding=embedding,
        )

        await repo.bulk_insert(doc.id, [row])

        chunks = await repo.list_for_document(doc.id)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_index == 7
        assert c.text == "specific text"
        assert c.page_number == 3
        assert c.section == "intro"
        assert c.token_count == 42
        assert c.document_id == doc.id
        assert isinstance(c.id, uuid.UUID)
        assert len(c.embedding) == settings.embedding_dim

    async def test_generated_ids_are_unique(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        rows = [make_chunk_row(chunk_index=i) for i in range(3)]

        await repo.bulk_insert(doc.id, rows)

        chunks = await repo.list_for_document(doc.id)
        ids = [c.id for c in chunks]
        assert len(set(ids)) == 3

    async def test_duplicate_chunk_index_raises_integrity_error(self, db: AsyncSession):
        """Relies on the uq_document_chunks_document_id_chunk_index constraint."""
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row(chunk_index=0)])

        with pytest.raises(IntegrityError):
            await repo.bulk_insert(doc.id, [make_chunk_row(chunk_index=0)])

    async def test_same_index_in_different_documents_is_allowed(self, db: AsyncSession):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        repo = ChunkRepository(db)

        await repo.bulk_insert(d1.id, [make_chunk_row(chunk_index=0)])
        await repo.bulk_insert(d2.id, [make_chunk_row(chunk_index=0)])

        assert await repo.count_for_document(d1.id) == 1
        assert await repo.count_for_document(d2.id) == 1

    async def test_text_search_column_is_populated_by_db(self, db: AsyncSession):
        """Computed TSVECTOR — should be filled on insert, not left NULL."""
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row(text="the quick brown fox")])

        chunk = (await repo.list_for_document(doc.id))[0]
        # TSVECTOR comes back as a string like "'brown':3 'fox':4 'quick':2 'the':1"
        assert chunk.text_search is not None
        assert "quick" in str(chunk.text_search)

    async def test_metadata_defaults_to_empty_dict(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row()])

        chunk = (await repo.list_for_document(doc.id))[0]
        assert chunk.metadata_ is not None
        assert isinstance(chunk.metadata_, dict)


# ─────────────────────────────────────────────────────────────
# delete_for_document
# ─────────────────────────────────────────────────────────────

class TestDeleteForDocument:
    async def test_returns_number_deleted(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row(chunk_index=i) for i in range(4)])

        deleted = await repo.delete_for_document(doc.id)

        assert deleted == 4
        assert await repo.count_for_document(doc.id) == 0

    async def test_returns_zero_when_no_chunks(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)

        assert await repo.delete_for_document(doc.id) == 0

    async def test_does_not_delete_other_documents_chunks(self, db: AsyncSession):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        repo = ChunkRepository(db)
        await repo.bulk_insert(d1.id, [make_chunk_row(chunk_index=0)])
        await repo.bulk_insert(d2.id, [make_chunk_row(chunk_index=0)])

        deleted = await repo.delete_for_document(d1.id)

        assert deleted == 1
        assert await repo.count_for_document(d1.id) == 0
        assert await repo.count_for_document(d2.id) == 1

    async def test_is_idempotent(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row()])

        assert await repo.delete_for_document(doc.id) == 1
        assert await repo.delete_for_document(doc.id) == 0


# ─────────────────────────────────────────────────────────────
# list_for_document
# ─────────────────────────────────────────────────────────────

class TestListForDocument:
    async def test_returns_in_chunk_index_order(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        # insert out of order on purpose
        await repo.bulk_insert(doc.id, [
            make_chunk_row(chunk_index=2),
            make_chunk_row(chunk_index=0),
            make_chunk_row(chunk_index=1),
        ])

        chunks = await repo.list_for_document(doc.id)

        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    async def test_empty_list_for_no_chunks(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)

        assert await repo.list_for_document(doc.id) == []

    async def test_does_not_leak_across_documents(self, db: AsyncSession):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        repo = ChunkRepository(db)
        await repo.bulk_insert(d1.id, [make_chunk_row(chunk_index=0, text="mine")])
        await repo.bulk_insert(d2.id, [make_chunk_row(chunk_index=0, text="theirs")])

        chunks = await repo.list_for_document(d1.id)

        assert len(chunks) == 1
        assert chunks[0].text == "mine"

    async def test_returns_document_chunk_instances(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row()])

        chunks = await repo.list_for_document(doc.id)

        assert all(isinstance(c, DocumentChunk) for c in chunks)


# ─────────────────────────────────────────────────────────────
# count_for_document
# ─────────────────────────────────────────────────────────────

class TestCountForDocument:
    async def test_counts_only_target_document(self, db: AsyncSession):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        repo = ChunkRepository(db)
        await repo.bulk_insert(d1.id, [make_chunk_row(chunk_index=i) for i in range(3)])
        await repo.bulk_insert(d2.id, [make_chunk_row(chunk_index=i) for i in range(7)])

        assert await repo.count_for_document(d1.id) == 3
        assert await repo.count_for_document(d2.id) == 7

    async def test_zero_for_missing_document(self, db: AsyncSession):
        repo = ChunkRepository(db)

        assert await repo.count_for_document(uuid.uuid4()) == 0


# ─────────────────────────────────────────────────────────────
# cascade behavior
# ─────────────────────────────────────────────────────────────

class TestCascade:
    async def test_deleting_document_cascades_to_chunks(self, db: AsyncSession):
        user = await _persist(db, make_user())
        doc = await _persist(db, make_document(user.id))
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row(chunk_index=i) for i in range(3)])

        await db.delete(doc)
        await db.flush()

        # bypass repository so we don't accidentally mask a cascade failure
        remaining = await db.scalar(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        )
        assert remaining is None


    async def test_delete_document_does_not_null_chunk_fks(self, db: AsyncSession):
        """If passive_deletes is missing, SQLAlchemy tries SET document_id=NULL
        and Postgres raises NotNullViolation. This test guards against that."""
        user = await _persist(db, make_user())
        doc = await _persist(db, make_document(user.id))
        repo = ChunkRepository(db)
        await repo.bulk_insert(doc.id, [make_chunk_row(chunk_index=i) for i in range(3)])

        # if this raises IntegrityError, passive_deletes=True is missing
        await db.delete(doc)
        await db.flush()
