"""Unit tests for CitationRepository."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.db.repositories.citation_repo import CitationRepository
from app.db.repositories.conversation_repo import ConversationRepository
from app.db.repositories.message_repo import MessageRepository
from app.models.enums import MessageRole
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.asyncio


@pytest.fixture
def repo(session):
    return CitationRepository(session)


@pytest.fixture
async def message(session, user):
    conv = await ConversationRepository(session).create(user_id=user.id)
    return await MessageRepository(session).create(conversation_id=conv.id, role=MessageRole.ASSISTANT, content="answer")


class TestBulkInsert:
    async def test_inserts_all_rows_and_returns_count(self, repo, message, chunk_factory):
        c1 = await chunk_factory()
        c2 = await chunk_factory()
        n = await repo.bulk_insert(
            [
                {"message_id": message.id, "chunk_id": c1.id, "score": 0.9, "rank": 1},
                {"message_id": message.id, "chunk_id": c2.id, "score": 0.8, "rank": 2},
            ]
        )
        assert n == 2

    async def test_empty_rows_returns_zero_and_does_not_touch_db(self, repo):
        assert await repo.bulk_insert([]) == 0

    async def test_rows_are_retrievable_with_all_fields(self, repo, message, chunk):
        await repo.bulk_insert([{"message_id": message.id, "chunk_id": chunk.id, "score": 0.75, "rank": 1}])
        rows = await repo.list_for_message(message.id)
        assert len(rows) == 1
        assert rows[0].score == 0.75
        assert rows[0].rank == 1
        assert rows[0].chunk_id == chunk.id

    async def test_duplicate_message_chunk_raises_integrity_error(self, repo, message, chunk):
        """One citation per (message, chunk); the second insert must fail."""
        row = {"message_id": message.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}
        await repo.bulk_insert([row])
        with pytest.raises(IntegrityError, match="uq_citation_message_chunk"):
            await repo.bulk_insert([row])


class TestListForMessage:
    async def test_returns_in_rank_order(self, repo, message, chunk_factory):
        c1 = await chunk_factory()
        c2 = await chunk_factory()
        c3 = await chunk_factory()
        await repo.bulk_insert(
            [
                {"message_id": message.id, "chunk_id": c1.id, "score": 0.1, "rank": 3},
                {"message_id": message.id, "chunk_id": c2.id, "score": 0.2, "rank": 1},
                {"message_id": message.id, "chunk_id": c3.id, "score": 0.3, "rank": 2},
            ]
        )
        rows = await repo.list_for_message(message.id)
        assert [r.rank for r in rows] == [1, 2, 3]

    async def test_empty_list_for_no_citations(self, repo, message):
        assert await repo.list_for_message(message.id) == []

    async def test_does_not_leak_across_messages(self, repo, session, user, message, chunk):
        other_conv = await ConversationRepository(session).create(user_id=user.id)
        other_msg = await MessageRepository(session).create(conversation_id=other_conv.id, role=MessageRole.ASSISTANT, content="other")
        await repo.bulk_insert([{"message_id": message.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}])
        await repo.bulk_insert([{"message_id": other_msg.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}])
        rows = await repo.list_for_message(message.id)
        assert [r.message_id for r in rows] == [message.id]


class TestCountForMessage:
    async def test_counts_only_target_message(self, repo, session, user, message, chunk):
        other_conv = await ConversationRepository(session).create(user_id=user.id)
        other_msg = await MessageRepository(session).create(conversation_id=other_conv.id, role=MessageRole.ASSISTANT, content="other")
        await repo.bulk_insert([{"message_id": message.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}])
        await repo.bulk_insert([{"message_id": other_msg.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}])
        assert await repo.count_for_message(message.id) == 1

    async def test_zero_for_missing_message(self, repo):
        assert await repo.count_for_message(uuid4()) == 0


class TestDeleteForMessage:
    async def test_returns_number_deleted(self, repo, message, chunk_factory):
        c1 = await chunk_factory()
        c2 = await chunk_factory()
        await repo.bulk_insert(
            [
                {"message_id": message.id, "chunk_id": c1.id, "score": 0.1, "rank": 1},
                {"message_id": message.id, "chunk_id": c2.id, "score": 0.2, "rank": 2},
            ]
        )
        assert await repo.delete_for_message(message.id) == 2

    async def test_is_idempotent(self, repo, message, chunk):
        await repo.bulk_insert([{"message_id": message.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}])
        await repo.delete_for_message(message.id)
        assert await repo.delete_for_message(message.id) == 0


class TestCascade:
    async def test_deleting_message_cascades_to_citations(self, repo, session, message, chunk):
        await repo.bulk_insert([{"message_id": message.id, "chunk_id": chunk.id, "score": 0.5, "rank": 1}])
        await session.delete(message)
        await session.flush()
        assert await repo.count_for_message(message.id) == 0
