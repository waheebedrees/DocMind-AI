from typing import Annotated
from uuid import UUID

from arq.connections import ArqRedis
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.arq import get_arq_pool
from app.core.auth import get_exception_401, get_subject_for_token_type
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.jobs import JobRepository
from app.db.repositories.user_repo import UserRepository
from app.db.session import get_db
from app.services.document_service import DocumentService
from app.services.storage import BaseStorage
from app.services.user_service import UserService


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

    _, email = get_subject_for_token_type(token, "access")
    return email


async def get_current_user_id(credentials: "Credentials") -> UUID:
    """
    FastAPI dependency to get current user ID from JWT token

    Validates JWT token and extracts user email.

    Args:
        credentials (Credentials): HTTP bearer token credentials (may be None if missing)

    Raise:
        HTTPException: if token is invalid or missing (401 Unauthorized)
    Returns:
        UUID: User ID

    """
    if credentials is None:
        raise get_exception_401("Unauthorized, messing authentication token")

    token = credentials.credentials
    if not token:
        raise get_exception_401("Unauthorized, empty authentication token")

    user_id, _ = get_subject_for_token_type(token, "access")
    try:
        return UUID(user_id)
    except Exception as exec:
        raise get_exception_401("Unauthorized") from exec


def get_user_repository(db: "DbSession") -> UserRepository:
    return UserRepository(db)


def get_user_service(user_repo: "UserRepositoryDep") -> UserService:
    "provide user service instance"
    return UserService(user_repo)


def get_storage(request: Request) -> BaseStorage:
    return request.app.state.storage


def get_document_repository(db: "DbSession") -> DocumentRepository:
    return DocumentRepository(session=db)


def get_chunk_repository(db: "DbSession") -> ChunkRepository:
    return ChunkRepository(session=db)


def get_job_repo(db: "DbSession") -> JobRepository:
    return JobRepository(session=db)


def get_document_service(document_repo: "DocumentRepositoryDep", job_repo: "JobRepositoryDep", chunk_repo: "ChunkRepositoryDep") -> DocumentService:
    return DocumentService(document_repo, job_repo=job_repo, chunk_repo=chunk_repo)


# shard http bearer across full app
security = HTTPBearer(auto_error=False)

Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(security)]
DbSession = Annotated[AsyncSession, Depends(get_db)]


CurrentUserEmail = Annotated[str, Depends(get_current_user_email)]
CurrentUserId = Annotated[UUID, Depends(get_current_user_id)]
StorageDep = Annotated[BaseStorage, Depends(get_storage)]
ArqPooleDep = Annotated[ArqRedis, Depends(get_arq_pool)]

UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]

JobRepositoryDep = Annotated[JobRepository, Depends(get_job_repo)]
DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]
ChunkRepositoryDep = Annotated[ChunkRepository, Depends(get_chunk_repository)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
