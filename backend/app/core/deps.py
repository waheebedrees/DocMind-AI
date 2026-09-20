from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_exception_401, get_subject_for_token_type
from app.db.repositories.user_repo import UserRepository
from app.db.session import get_db
from app.services.user_service import UserService

# shard http bearer across full app
security = HTTPBearer(auto_error=False)


async def get_current_user_email(credentials: "Credentials") -> str:
    """
    FastAPI dependency to get current user email from JWT token

    Validates JWT token and extracts user email.

    Args:
        credentials (Credentials): HTTP bearer token credentials (may be None if missing)

    Raise:
        HTTPException: if token is invalid or missing (401 Unauthorized)
    Returns:
        str: User email address

    Example:
        ```python
        @router.get('profile')
        async def get_profile(
            email: Annotated[str, Depends(get_current_user_email]
        ):
            # email is the authorized user's email
        ```
    """
    if credentials is None:
        detail = "Unauthorized, messing authentication token"
        raise get_exception_401(detail)

    token = credentials.credentials
    if not token:
        detail = "Unauthorized, empty authentication token"
        raise get_exception_401(detail)

    user_id, email = get_subject_for_token_type(token, "access")
    return email


def get_user_repository(db: "DbSession") -> UserRepository:
    "provide user repository instance"
    return UserRepository(db)


def get_user_service(user_repo: "UserRepo") -> UserService:
    return UserService(user_repo)


Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(security)]

DbSession = Annotated[AsyncSession, Depends(get_db)]
UserRepo = Annotated[UserRepository, Depends(get_user_repository)]
