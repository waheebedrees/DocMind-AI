"""Unit tests for MessageRepository."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.models.enums import MessageRole
from app.db.repositories.conversation_repo import ConversationRepository
from app.db.repositories.message_repo import MessageRepository

pytestmark = pytest.mark.asyncio


@pytest.fixture
def repo(session):
    return MessageRepository(session)


@pytest.fixture
async def conversation(session, user):
    return await ConversationRepository(session).create(user_id=user.id)


class TestCreate:
    async def test_persists_all_fields(self, repo, session, conversation):
        msg = await repo.create(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="hello",
            tokens_in=10,
            tokens_out=20,
            cost_usd=Decimal("0.001234"),
            latency_ms=42,
        )
        await session.refresh(msg)
        assert msg.id is not None
        assert msg.conversation_id == conversation.id
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"
        assert msg.tokens_in == 10
        assert msg.tokens_out == 20
        assert msg.cost_usd == Decimal("0.001234")
        assert msg.latency_ms == 42

    async def test_optional_fields_default_to_none(self, repo, conversation):
        msg = await repo.create(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content="hi",
        )
        assert msg.tokens_in is None
        assert msg.tokens_out is None
        assert msg.cost_usd is None
        assert msg.latency_ms is None


class TestGet:
    async def test_returns_message_when_found(self, repo, conversation):
        msg = await repo.create(
            conversation_id=conversation.id, role=MessageRole.USER, content="x"
        )
        got = await repo.get(msg.id)
        assert got is not None
        assert got.id == msg.id

    async def test_returns_none_when_missing(self, repo):
        from uuid import uuid4

        assert await repo.get(uuid4()) is None


class TestListForConversation:
    async def test_returns_in_creation_order(self, repo, conversation):
        a = await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content="a")
        b = await repo.create(conversation_id=conversation.id, role=MessageRole.ASSISTANT, content="b")
        c = await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content="c")
        rows = await repo.list_for_conversation(conversation.id)
        assert [m.id for m in rows] == [a.id, b.id, c.id]

    async def test_empty_for_conversation_with_no_messages(self, repo, conversation):
        assert await repo.list_for_conversation(conversation.id) == []

    async def test_does_not_leak_across_conversations(self, repo, session, user, conversation):
        other = await ConversationRepository(session).create(user_id=user.id)
        await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content="mine")
        await repo.create(conversation_id=other.id, role=MessageRole.USER, content="theirs")
        rows = await repo.list_for_conversation(conversation.id)
        assert [m.content for m in rows] == ["mine"]


class TestLatestForConversation:
    async def test_returns_n_newest_oldest_first(self, repo, conversation):
        msgs = [
            await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content=str(i))
            for i in range(5)
        ]
        rows = await repo.latest_for_conversation(conversation.id, limit=3)
        assert [m.id for m in rows] == [msgs[2].id, msgs[3].id, msgs[4].id]

    async def test_respects_limit(self, repo, conversation):
        for i in range(10):
            await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content=str(i))
        rows = await repo.latest_for_conversation(conversation.id, limit=4)
        assert len(rows) == 4

    async def test_empty_when_no_messages(self, repo, conversation):
        assert await repo.latest_for_conversation(conversation.id) == []


class TestCountForConversation:
    async def test_counts_only_target_conversation(self, repo, session, user, conversation):
        other = await ConversationRepository(session).create(user_id=user.id)
        await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content="a")
        await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content="b")
        await repo.create(conversation_id=other.id, role=MessageRole.USER, content="c")
        assert await repo.count_for_conversation(conversation.id) == 2

    async def test_zero_for_missing(self, repo):
        from uuid import uuid4

        assert await repo.count_for_conversation(uuid4()) == 0


class TestDeleteForConversation:
    async def test_returns_number_deleted(self, repo, conversation):
        for i in range(3):
            await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content=str(i))
        assert await repo.delete_for_conversation(conversation.id) == 3

    async def test_is_idempotent(self, repo, conversation):
        await repo.create(conversation_id=conversation.id, role=MessageRole.USER, content="x")
        await repo.delete_for_conversation(conversation.id)
        assert await repo.delete_for_conversation(conversation.id) == 0
