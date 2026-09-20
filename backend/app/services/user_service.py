from uuid import UUID

from app.core.auth import (
    create_token_pair,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.core.logging import get_logger
from app.db.repositories.user_repo import UserRepository
from app.models import User
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserResponse

log = get_logger(__name__)


class UserService:
    """
    service for user management operations
    handle business logic for user
    """

    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    async def get_by_id(self, user_id: UUID) -> UserResponse | None:
        """
        get user by id

        Args:
            user_id (UUID): primary user id

        Returns:
            UserResponse or None if not found
        """
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return None
        return _convert_user_to_schema(user)

    async def get_user_by_email(self, email: str) -> UserResponse | None:
        """
        get user by email

        Args:
            email (str): user email

        Returns:
            UserResponse or None if not found
        """

        user = await self.user_repo.get_by_email(email)
        if not user:
            return None
        return _convert_user_to_schema(user)

    async def get_all_user(self, skip: int = 0, limit: int = 100) -> list[UserResponse]:
        """
        get all user with pagination

        Args:
            skip (int, optional): number of recodes to skip. Defaults to 0.
            limit (int, optional): maximum number of recodes. Defaults to 100.

        Returns:
            list of UserResponse instance
        """

        users = await self.user_repo.get_all(skip=skip, limit=limit)
        return [_convert_user_to_schema(user) for user in users]

    async def register_user(
        self,
        user_data: UserCreate,
    ) -> UserResponse:
        """
        register new user with email

        Args:
            user_data (UserCreate): user creation data
        """
        log.info("user registration", extra={"email": user_data.email})

        if await self.user_repo.exists_by_email(email=user_data.email):
            log.warning("user registration failed")
            raise ValueError("Email Already registered")

        hashed_password = hash_password(user_data.password)

        user = await self.user_repo.create_user(
            email=user_data.email,
            hash_password=hashed_password,
            is_active=True,
            is_verified=False,
        )

        return _convert_user_to_schema(user)

    async def authenticate_user(self, email: str, password: str) -> Token:

        user = await self.user_repo.get_by_email(email)
        if not user:
            log.warning("user authentication failed")
            raise ValueError("there are no user with current email")

        if not verify_password(password, user.password_hash):
            log.warning("invalid email or password")
            raise ValueError("Invalid email or password")

        return create_token_pair(user_id=str(user.id), email=user.email)

    async def refresh(self, token: str) -> Token:
        payload = decode_refresh_token(token)  # validates type, jti, exp

        try:
            user_id = UUID(payload["sub"])
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid token subject") from e

        user = await self.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        if not user.is_active:
            raise ValueError("User is inactive")

        # Optional, if you require verified users
        # if not user.is_verified:
        #     raise ValueError("User is not verified")

        return create_token_pair(user_id=str(user.id), email=user.email)


def _convert_user_to_schema(user: User) -> UserResponse:
    return UserResponse.model_validate(user)
