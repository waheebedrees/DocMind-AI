from fastapi import APIRouter, Body, Depends, Request, status

from app.core.auth import get_exception_401, get_exception_409
from app.core.deps import get_user_service
from app.core.logging import get_logger
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserLogin, UserResponse
from app.services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["Authentication"])

log = get_logger(__name__)


@router.post(
    "/refresh",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
    description="Use a valid refresh token to obtain a new access token (and new refresh token).",
    responses={
        200: {"description": "Token refreshed successfully"},
        401: {"description": "Invalid or expired refresh token"},
    },
)
async def refresh_token(
    refresh_token: str = Body(..., embed=True),
    user_service: UserService = Depends(get_user_service),
) -> Token:
    try:
        return await user_service.refresh(refresh_token)
    except ValueError as e:
        log.warning("Refresh failed", extra={"error": str(e)})
        raise get_exception_401("Invalid or expired refresh token") from e


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(request: Request, body: UserCreate, user_service: UserService = Depends(get_user_service)):
    try:
        user = await user_service.register_user(body)
        return user
    except ValueError as e:
        raise get_exception_409(str(e)) from e


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="user login",
    responses={
        200: {"description": "authentication successfully"},
        401: {"description": "Invalid credentials "},
    },
)
async def login(request: Request, body: UserLogin, user_service: UserService = Depends(get_user_service)):
    try:
        return await user_service.authenticate_user(body.email, body.password)
    except ValueError as e:
        raise get_exception_401("invalid email or password") from e
