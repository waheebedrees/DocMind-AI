from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.user import User


class UserRepository(BaseRepository[User]):
    """
    repository for user model with domain-specific queries
    """

    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        """
        get user by email

        Args:
            email (str): user email

        Returns:
            Optional[User]: user instance if exists None if not found
        """

        stmt = select(User).where(User.email == email)
        user = await self.session.execute(stmt)

        return user.scalar_one_or_none()

    async def exists_by_email(self, email: str) -> bool:
        """
        check if the user exists by email

        Args:
            email (str): user email

        Returns:
            bool: True if use exists False if otherwise
        """

        user = await self.get_by_email(email)
        return user is not None

    async def create_user(self, email: str, hash_password: str, is_active: bool = False, is_verified: bool = False) -> User:
        """
        create user recode

        Args:
            email (str): user email
            hash_password (str): user hashed password
            username (str): user name
            is_active (bool, optional): whether user active. Defaults to False.
            is_verified (bool, optional): whether user email verified or not. Defaults to False.

        Returns:
            created user instance
        """

        user = User(email=email, password_hash=hash_password, is_active=is_active, is_verified=is_verified)

        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def confirm_user_email(
        self,
        user_id: UUID,
    ) -> User | None:
        """
        confirm user email and activate account

        Args:
            user_id (UUID): user id to confirm

        Returns:
            Optional[User]: updated user instance or None if not found
        """
        return await self.update(id=user_id, is_verified=True, is_active=True)
