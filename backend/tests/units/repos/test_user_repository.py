import uuid as _uuid

import pytest
from app.db.repositories.user_repo import UserRepository
from app.models.user import User
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


def unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{_uuid.uuid4().hex[:10]}@example.com"


def make_user(
    email="ewaheeb02@gmail.com",
    password_hash="password_hash",
    is_active=True,
    is_verified=True,
):
    return User(
        email=email,
        password_hash=password_hash,
        is_active=is_active,
        is_verified=is_verified,
    )


pytestmark = pytest.mark.unit


async def _persist(db: AsyncSession, *objs):
    for o in objs:
        db.add(o)
    await db.flush()
    for o in objs:
        await db.refresh(o)
    return objs if len(objs) > 1 else objs[0]


class TestGetByEmail:
    async def test_returns_user_when_found(self, db: AsyncSession):
        email = unique_email()
        await _persist(db, make_user(email=email))
        repo = UserRepository(db)

        user = await repo.get_by_email(email)

        assert user is not None
        assert user.email == email

    async def test_returns_none_when_missing(self, db: AsyncSession):
        repo = UserRepository(db)

        assert await repo.get_by_email(unique_email("nobody")) is None

    async def test_returns_none_when_email_is_empty(self, db: AsyncSession):
        repo = UserRepository(db)

        assert await repo.get_by_email("") is None


class TestExistsByEmail:
    async def test_true_when_present(self, db: AsyncSession):
        email = unique_email()
        await _persist(db, make_user(email=email))
        repo = UserRepository(db)

        assert await repo.exists_by_email(email) is True

    async def test_false_when_missing(self, db: AsyncSession):
        repo = UserRepository(db)

        assert await repo.exists_by_email(unique_email("ghost")) is False


class TestCreateUser:
    async def test_persists_all_fields(self, db: AsyncSession):
        """Catches the missing-username bug."""
        repo = UserRepository(db)
        email = unique_email()

        user = await repo.create_user(
            email=email,
            hash_password="hashed-value",
        )

        assert user.id is not None
        assert user.email == email
        assert user.password_hash == "hashed-value"
        assert user.is_active is False
        assert user.is_verified is False

    async def test_respects_explicit_flags(self, db: AsyncSession):
        repo = UserRepository(db)

        user = await repo.create_user(
            email=unique_email(),
            hash_password="x",
            is_active=True,
            is_verified=True,
        )

        assert user.is_active is True
        assert user.is_verified is True

    async def test_row_is_queryable_from_same_session(self, db: AsyncSession):
        """create_user must leave the new row visible without an explicit commit."""
        repo = UserRepository(db)
        email = unique_email()

        user = await repo.create_user(email=email, hash_password="x")
        found = await repo.get_by_email(email)

        assert found is not None
        assert found.id == user.id

    async def test_duplicate_email_raises(self, db: AsyncSession):
        """Relies on a UNIQUE constraint on users.email."""
        repo = UserRepository(db)
        email = unique_email()

        await repo.create_user(email=email, hash_password="x")

        with pytest.raises(IntegrityError):
            await repo.create_user(email=email, hash_password="y")

    async def test_returns_distinct_ids_for_distinct_users(self, db: AsyncSession):
        repo = UserRepository(db)

        u1 = await repo.create_user(email=unique_email("a"), hash_password="x")
        u2 = await repo.create_user(email=unique_email("b"), hash_password="y")

        assert u1.id != u2.id

    async def test_does_not_commit_behind_caller_back(self, db: AsyncSession):
        """
        Guard against a commit() sneaking back in: after create_user, an outer
        rollback must remove the row. If this fails, create_user is committing.
        """
        repo = UserRepository(db)
        email = unique_email()

        await repo.create_user(email=email, hash_password="x")
        await db.rollback()

        assert await repo.get_by_email(email) is None


class TestConfirmUserEmail:
    async def test_sets_both_flags(self, db: AsyncSession):
        user = await _persist(db, make_user(email=unique_email(), is_active=False, is_verified=False))
        repo = UserRepository(db)

        updated = await repo.confirm_user_email(user.id)

        assert updated is not None
        assert updated.is_verified is True
        assert updated.is_active is True

    async def test_returns_none_when_missing(self, db: AsyncSession):
        repo = UserRepository(db)

        result = await repo.confirm_user_email(_uuid.uuid4())

        assert result is None

    async def test_is_idempotent(self, db: AsyncSession):
        user = await _persist(db, make_user(email=unique_email(), is_active=True, is_verified=True))
        repo = UserRepository(db)

        first = await repo.confirm_user_email(user.id)
        second = await repo.confirm_user_email(user.id)

        assert first is not None and second is not None
        assert first.id == second.id
        assert first.is_verified is True
        assert first.is_active is True

    async def test_persisted_state_matches_return_value(self, db: AsyncSession):
        user = await _persist(db, make_user(email=unique_email()))
        repo = UserRepository(db)

        updated = await repo.confirm_user_email(user.id)
        await db.refresh(updated)

        assert updated.is_active is True
        assert updated.is_verified is True
