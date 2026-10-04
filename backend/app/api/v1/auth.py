"""Authentication routes.

Four endpoints under ``/auth``:

* ``POST /register`` — create a user.
* ``POST /login`` — exchange email + password for a token pair.
* ``POST /refresh`` — exchange a refresh token for a new token pair.
* ``GET  /me`` — return the authenticated user's profile.

All error handling here is a thin adapter: the service layer raises
``ValueError`` with a human-readable message, and each endpoint maps
that to the appropriate HTTP status. The service does not know about
HTTP; this module does not know about password hashing, token signing,
or user persistence.
"""

from fastapi import APIRouter, Body, Request, status

from app.core.auth import get_exception_401, get_exception_409
from app.core.deps import CurrentUserEmail, UserServiceDep
from app.core.logging import get_logger
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserLogin, UserResponse

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
    user_service: UserServiceDep,
    refresh_token: str = Body(..., embed=True),
) -> Token:
    """Exchange a refresh token for a new token pair.

    Rotation semantics are up to the service: the returned ``Token``
    contains both a new access token and a new refresh token, and the
    caller should discard the old refresh token. The endpoint does not
    revoke the old token itself — if the service does not either, replay
    is possible.

    Args:
        user_service: Injected user service. Owns token verification
            and issuance.
        refresh_token: The refresh token from a previous login or
            refresh, sent as ``{"refresh_token": "..."}``. Embedded so
            the body is an object rather than a bare string.

    Returns:
        A ``Token`` with the new access and refresh tokens.

    Raises:
        HTTPException: 401 if the refresh token is invalid, expired, or
            has been revoked. The underlying ``ValueError`` from the
            service is logged at WARNING and not surfaced to the client
            — the response intentionally does not distinguish failure
            modes to avoid leaking which tokens were once valid.
    """
    try:
        return await user_service.refresh(refresh_token)
    except ValueError as e:
        log.warning("Refresh failed", extra={"error": str(e)})
        raise get_exception_401("Invalid or expired refresh token") from e


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(request: Request, body: UserCreate, user_service: UserServiceDep):
    """Create a new user account.

    The password in ``body`` is hashed by the service before
    persistence; it never appears in logs, metrics, or the response.

    Args:
        request: The incoming request. Present for symmetry with the
            other endpoints and available for future use (e.g. rate
            limiting by client IP); not currently read.
        body: The new user's email and password.
        user_service: Injected user service. Owns uniqueness checks and
            password hashing.

    Returns:
        The created user's public representation. Does not include the
        password hash.

    Raises:
        HTTPException: 409 if the email is already registered. The
            service's ``ValueError`` message is passed through as the
            response detail — it is assumed to be user-safe.
    """
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
async def login(request: Request, body: UserLogin, user_service: UserServiceDep):
    """Authenticate an email + password and issue a token pair.

    Args:
        request: The incoming request. Not read by this handler;
            present for symmetry and future use (client fingerprinting,
            failed-login rate limiting).
        body: The user's email and password.
        user_service: Injected user service. Owns credential
            verification and token issuance.

    Returns:
        A ``Token`` with a fresh access token and refresh token.

    Raises:
        HTTPException: 401 for *any* authentication failure — bad
            email, bad password, locked account, unverified email. The
            service's original message is discarded and replaced with
            the generic "invalid email or password" so an attacker
            cannot distinguish "no such user" from "wrong password" by
            response body.
    """
    try:
        return await user_service.authenticate_user(body.email, body.password)
    except ValueError as e:
        raise get_exception_401("invalid email or password") from e


@router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    summary="get current user info",
    responses={
        200: {"description": " current user info"},
        401: {"description": "Invalid email"},
    },
)
async def me(email: CurrentUserEmail, user_service: UserServiceDep):
    """Return the authenticated user's profile.

    The caller's identity is established by the authentication
    dependency (``CurrentUserEmail``), which validates the bearer token
    before this handler runs. The email parameter is not client input —
    it is the value the dependency extracted from a verified token.

    Args:
        email: The authenticated user's email, injected by the auth
            dependency. Type is a FastAPI dependency alias, not a plain
            string, so a missing or invalid token fails before this
            function is invoked.
        user_service: Injected user service.

    Returns:
        The user's public representation.

    Raises:
        HTTPException: 401 if no user matches the token's email. In
            practice this means the user was deleted after the token
            was issued but before it expired; the token is still
            cryptographically valid, but the account is gone.
    """
    try:
        return await user_service.get_user_by_email(email)
    except ValueError as e:
        raise get_exception_401("invalid user") from e
