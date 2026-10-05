"""Unit tests for ConversationRepository."""

from __future__ import annotations

import pytest

from app.db.repositories.conversation_repo import ConversationRepository

pytestmark = pytest.mark.asyncio


@pytest.fixture
def repo(session):
    return ConversationRepository(session)


class TestCreate:
    async def test_persists_all_fields(self, repo, session, user):
        conv = await repo.create(user_id=user.id, title="first")
        await session.refresh(conv)
        assert conv.id is not None
        assert conv.user_id == user.id
        assert conv.title == "first"
        assert conv.created_at is not None

    async def test_title_defaults_to_none(self, repo, user):
        conv = await repo.create(user_id=user.id)
        assert conv.title is None

    async def test_two_conversations_get_distinct_ids(self, repo, user):
        a = await repo.create(user_id=user.id)
        b = await repo.create(user_id=user.id)
        assert a.id != b.id


class TestGetForUser:
    async def test_returns_conversation_when_owner_matches(self, repo, user):
        conv = await repo.create(user_id=user.id, title="t")
        got = await repo.get_for_user(conv.id, user.id)
        assert got is not None
        assert got.id == conv.id

    async def test_returns_none_when_wrong_user(self, repo, user, other_user):
        conv = await repo.create(user_id=user.id)
        assert await repo.get_for_user(conv.id, other_user.id) is None

    async def test_returns_none_when_conversation_missing(self, repo, user):
        from uuid import uuid4

        assert await repo.get_for_user(uuid4(), user.id) is None


class TestListForUser:

    async def test_returns_newest_first(self, repo, session, user):
            from datetime import UTC, datetime, timedelta

            base = datetime(2024, 1, 1, tzinfo=UTC)
            first = await repo.create(user_id=user.id, title="a")
            second = await repo.create(user_id=user.id, title="b")
            first.updated_at = base
            second.updated_at = base + timedelta(seconds=1)
            await session.flush()

            rows = await repo.list_for_user(user.id)
            assert [c.id for c in rows] == [second.id, first.id]
            
    async def test_pagination_limit_and_offset(self, repo, user):
        convs = [await repo.create(user_id=user.id, title=str(i)) for i in range(5)]
        page = await repo.list_for_user(user.id, limit=2, offset=1)
        assert len(page) == 2
        # whatever the ordering, page ⊆ convs
        assert {c.id for c in page} <= {c.id for c in convs}

    async def test_empty_for_user_with_no_conversations(self, repo, user):
        assert await repo.list_for_user(user.id) == []

    async def test_does_not_return_other_users_conversations(self, repo, user, other_user):
        mine = await repo.create(user_id=user.id)
        await repo.create(user_id=other_user.id)
        rows = await repo.list_for_user(user.id)
        assert [c.id for c in rows] == [mine.id]


class TestCountForUser:
    async def test_counts_only_target_user(self, repo, user, other_user):
        await repo.create(user_id=user.id)
        await repo.create(user_id=user.id)
        await repo.create(user_id=other_user.id)
        assert await repo.count_for_user(user.id) == 2

    async def test_zero_when_empty(self, repo, user):
        assert await repo.count_for_user(user.id) == 0


class TestSetTitle:
    async def test_updates_title(self, repo, user):
        conv = await repo.create(user_id=user.id, title="old")
        await repo.set_title(conv.id, "new")
        assert conv.title == "new"

    async def test_can_clear_title(self, repo, user):
        conv = await repo.create(user_id=user.id, title="x")
        await repo.set_title(conv.id, None)
        assert conv.title is None

    async def test_missing_conversation_raises(self, repo):
        from uuid import uuid4

        with pytest.raises(LookupError):
            await repo.set_title(uuid4(), "x")


class TestDeleteForUser:
    async def test_returns_one_when_deleted(self, repo, user):
        conv = await repo.create(user_id=user.id)
        assert await repo.delete_for_user(conv.id, user.id) == 1

    async def test_does_not_delete_other_users_conversation(self, repo, user, other_user):
        conv = await repo.create(user_id=other_user.id)
        assert await repo.delete_for_user(conv.id, user.id) == 0
        assert await repo.get_for_user(conv.id, other_user.id) is not None

    async def test_is_idempotent(self, repo, user):
        conv = await repo.create(user_id=user.id)
        assert await repo.delete_for_user(conv.id, user.id) == 1
        assert await repo.delete_for_user(conv.id, user.id) == 0
