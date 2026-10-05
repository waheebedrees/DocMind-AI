"""Unit tests for ExtractionRepository."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.repositories.extraction_repo import ExtractionRepository

pytestmark = pytest.mark.asyncio


@pytest.fixture
def repo(session):
    return ExtractionRepository(session)


class TestCreate:
    async def test_persists_all_fields(self, repo, session, document):
        row = await repo.create(
            document_id=document.id,
            schema_name="invoice",
            data={"total": 100},
            confidence=0.9,
            needs_review=True,
        )
        await session.refresh(row)
        assert row.id is not None
        assert row.document_id == document.id
        assert row.schema_name == "invoice"
        assert row.data == {"total": 100}
        assert row.confidence == 0.9
        assert row.needs_review is True

    async def test_needs_review_defaults_false(self, repo, document):
        row = await repo.create(
            document_id=document.id,
            schema_name="invoice",
            data={},
            confidence=0.5,
        )
        assert row.needs_review is False

    async def test_confidence_out_of_range_raises(self, repo, document):
        with pytest.raises(IntegrityError):
            await repo.create(
                document_id=document.id,
                schema_name="invoice",
                data={},
                confidence=1.5,
            )


class TestGet:
    async def test_returns_extraction_when_found(self, repo, document):
        row = await repo.create(
            document_id=document.id, schema_name="s", data={}, confidence=0.5
        )
        assert (await repo.get(row.id)).id == row.id

    async def test_returns_none_when_missing(self, repo):
        from uuid import uuid4

        assert await repo.get(uuid4()) is None


class TestListForDocument:
    async def test_returns_newest_first(self, repo, session, document):
        import asyncio

        a = await repo.create(document_id=document.id, schema_name="s", data={}, confidence=0.5)
        await asyncio.sleep(0.001)
        b = await repo.create(document_id=document.id, schema_name="s", data={}, confidence=0.5)
        rows = await repo.list_for_document(document.id)
        assert [r.id for r in rows] == [b.id, a.id]

    async def test_does_not_leak_across_documents(self, repo, session, document, other_document):
        await repo.create(document_id=document.id, schema_name="s", data={}, confidence=0.5)
        await repo.create(document_id=other_document.id, schema_name="s", data={}, confidence=0.5)
        rows = await repo.list_for_document(document.id)
        assert all(r.document_id == document.id for r in rows)

    async def test_empty_for_document_with_no_extractions(self, repo, document):
        assert await repo.list_for_document(document.id) == []


class TestGetForDocumentBySchema:
    async def test_returns_newest_for_schema(self, repo, session, document):
        import asyncio

        await repo.create(document_id=document.id, schema_name="invoice", data={"v": 1}, confidence=0.5)
        await asyncio.sleep(0.001)
        newest = await repo.create(
            document_id=document.id, schema_name="invoice", data={"v": 2}, confidence=0.5
        )
        got = await repo.get_for_document_by_schema(document.id, "invoice")
        assert got.id == newest.id

    async def test_returns_none_for_unknown_schema(self, repo, document):
        await repo.create(document_id=document.id, schema_name="invoice", data={}, confidence=0.5)
        assert await repo.get_for_document_by_schema(document.id, "receipt") is None


class TestListNeedingReview:
    async def test_returns_only_flagged(self, repo, document):
        flagged = await repo.create(
            document_id=document.id, schema_name="s", data={}, confidence=0.1, needs_review=True
        )
        await repo.create(
            document_id=document.id, schema_name="s", data={}, confidence=0.9, needs_review=False
        )
        rows = await repo.list_needing_review()
        assert [r.id for r in rows] == [flagged.id]

    async def test_can_filter_to_one_document(self, repo, document, other_document):
        await repo.create(
            document_id=document.id, schema_name="s", data={}, confidence=0.1, needs_review=True
        )
        await repo.create(
            document_id=other_document.id, schema_name="s", data={}, confidence=0.1, needs_review=True
        )
        rows = await repo.list_needing_review(document_id=document.id)
        assert all(r.document_id == document.id for r in rows)


class TestSetNeedsReview:
    async def test_flips_flag(self, repo, document):
        row = await repo.create(
            document_id=document.id, schema_name="s", data={}, confidence=0.5
        )
        await repo.set_needs_review(row.id, True)
        assert row.needs_review is True
        await repo.set_needs_review(row.id, False)
        assert row.needs_review is False

    async def test_missing_returns_none(self, repo):
        from uuid import uuid4

        assert await repo.set_needs_review(uuid4(), True) is None


class TestDeleteForDocument:
    async def test_returns_number_deleted(self, repo, document):
        for _ in range(3):
            await repo.create(document_id=document.id, schema_name="s", data={}, confidence=0.5)
        assert await repo.delete_for_document(document.id) == 3

    async def test_is_idempotent(self, repo, document):
        await repo.create(document_id=document.id, schema_name="s", data={}, confidence=0.5)
        await repo.delete_for_document(document.id)
        assert await repo.delete_for_document(document.id) == 0
