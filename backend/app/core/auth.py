"""Authentication utilities for JWT token and password handling.

This module provides helpers for:
- Password hashing and verification using bcrypt.
- JWT access/refresh token creation and validation.
- Standardized HTTP exceptions for auth-related errors.
"""

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
import jwt
from fastapi import HTTPException, status
from jwt import ExpiredSignatureError, PyJWTError

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.auth import Token

log = get_logger(__name__)


def _jwt_secret() -> str:
    """Return the configured JWT secret.

    Returns:
        str: The JWT signing secret.

    Raises:
        RuntimeError: If `settings.jwt_secret` is not configured.
    """
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured")
    return settings.jwt_secret


def _jwt_algorithm() -> str:
    """Return the configured JWT signing algorithm.

    Returns:
        str: The signing algorithm, defaulting to ``"HS256"``.
    """
    return settings.jwt_algorithm or "HS256"


def get_exception_400(detail: str) -> HTTPException:
    """Build an HTTP 400 Bad Request exception.

    Args:
        detail: Human-readable error message.

    Returns:
        HTTPException: Exception with status code 400.
    """
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def get_exception_401(detail: str) -> HTTPException:
    """Build an HTTP 401 Unauthorized exception.

    Includes the ``WWW-Authenticate: Bearer`` header as required by RFC 6750.

    Args:
        detail: Human-readable error message.

    Returns:
        HTTPException: Exception with status code 401.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_exception_403(detail: str) -> HTTPException:
    """Build an HTTP 403 Forbidden exception.

    Args:
        detail: Human-readable error message.

    Returns:
        HTTPException: Exception with status code 403.
    """
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_exception_409(detail: str) -> HTTPException:
    """Build an HTTP 409 Conflict exception.

    Args:
        detail: Human-readable error message.

    Returns:
        HTTPException: Exception with status code 409.
    """
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt.

    Args:
        password: The plain-text password to hash.

    Returns:
        str: The bcrypt hash, UTF-8 decoded.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash.

    Args:
        plain: The plain-text password to check.
        hashed: The stored bcrypt hash.

    Returns:
        bool: ``True`` if the password matches, ``False`` otherwise.
    """
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str) -> str:
    """Create a short-lived JWT access token.

    Args:
        user_id: The subject (``sub``) claim — typically the user's UUID.
        email: The user's email, embedded in the ``email`` claim.

    Returns:
        str: The encoded JWT access token.
    """
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    access_token_payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(UTC).timestamp(),
    }

    return jwt.encode(access_token_payload, _jwt_secret(), algorithm=_jwt_algorithm())


def create_refresh_token(user_id: str, email: str) -> tuple[str, datetime]:
    """Create a long-lived JWT refresh token.

    Args:
        user_id: The subject (``sub``) claim — typically the user's UUID.
        email: The user's email, embedded in the ``email`` claim.

    Returns:
        tuple[str, datetime]: A pair of ``(token, expires_at)`` where
        ``expires_at`` is the token's UTC expiration timestamp.
    """
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    refresh_token_payload = {
        "sub": user_id,
        "email": email,
        "type": "refresh",
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(UTC).timestamp(),
    }

    return jwt.encode(refresh_token_payload, _jwt_secret(), algorithm=_jwt_algorithm()), expire


def create_token_pair(user_id: str, email: str) -> Token:
    """Create both an access token and a refresh token.

    Args:
        user_id: The subject (``sub``) claim — typically the user's UUID.
        email: The user's email, embedded in the ``email`` claim.

    Returns:
        Token: Schema populated with both tokens, the access token's
        lifetime in seconds, and the refresh token's expiration datetime.
    """
    access_token = create_access_token(user_id, email)
    refresh_token, refresh_expires_at = create_refresh_token(user_id, email)
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
        refresh_expires_at=refresh_expires_at,
    )


def get_subject_for_token_type(
    token: str,
    payload_type: Literal["access", "refresh"],
) -> tuple[str, str]:
    """Decode a token and return its subject, validating the ``type`` claim.

    Args:
        token: The encoded JWT to decode.
        payload_type: The expected value of the token's ``type`` claim,
            either ``"access"`` or ``"refresh"``.

    Returns:
        tuple[str, str]: A pair of ``(user_id, email)``.

    Raises:
        HTTPException: 401 if the token is invalid, expired, missing
            required claims, or has a mismatched ``type`` claim.
    """
    payload = decode_token_payload(token)

    user_id = payload.get("sub")
    email = payload.get("email")
    token_type = payload.get("type")

    if not user_id or not email:
        raise get_exception_401("Unauthorized, malformed token")

    if token_type != payload_type:
        log.warning(
            "Token type mismatch",
            extra={"token_type": token_type, "expected": payload_type},
        )
        raise get_exception_401("Unauthorized, invalid token type")

    return user_id, email


def decode_token_payload(token: str) -> dict[str, Any]:
    """Decode a JWT token and return its payload.

    Signature and expiration are verified; the ``type`` claim is not.

    Args:
        token: The encoded JWT to decode.

    Returns:
        dict[str, Any]: The decoded token payload.

    Raises:
        HTTPException: 401 if the token is expired or otherwise invalid.

    Example:
        ```python
        payload = decode_token_payload(token)
        expires_at = payload.get("exp")
        ```
    """
    try:
        payload = jwt.decode(token, key=_jwt_secret(), algorithm=_jwt_algorithm())
        return payload
    except ExpiredSignatureError as e:
        detail = "Unauthorized, Token has expired"
        log.warning("Token expired", extra={"error": str(e)})
        raise get_exception_401(detail) from e
    except PyJWTError as e:
        detail = "Unauthorized, invalid token"
        log.warning("Token invalid", extra={"error": str(e)})
        raise get_exception_401(detail) from e


def decode_refresh_token(token: str) -> dict[str, Any]:
    """Decode and validate a refresh token.

    Args:
        token: The encoded refresh JWT.

    Returns:
        dict[str, Any]: The decoded token payload.

    Raises:
        HTTPException: 401 if the token is invalid, expired, not of type
            ``"refresh"``, or missing its ``jti`` claim.
    """
    payload = decode_token_payload(token)  # already raises 401 on invalid/expired
    if payload.get("type") != "refresh":
        raise get_exception_401("Unauthorized, invalid token type")
    if not payload.get("jti"):
        raise get_exception_401("Unauthorized, malformed refresh token")
    return payload


def verify_token(token: str) -> tuple[str, str] | None:
    """Verify a JWT token and extract its subject and email.

    Unlike :func:`decode_token_payload`, this function performs a
    lightweight shape check before decoding and returns ``None`` instead
    of raising on invalid tokens. It still raises ``ValueError`` for
    clearly malformed input (empty or non-string).

    Args:
        token: The encoded JWT to verify.

    Returns:
        tuple[str, str] | None: ``(user_id, email)`` on success, or
        ``None`` if the token is invalid, expired, or missing required
        claims.

    Raises:
        ValueError: If ``token`` is not a non-empty string or does not
            resemble a JWT (``header.payload.signature``).
    """
    if not token or not isinstance(token, str):
        log.warning("Token invalid format")
        raise ValueError("Token must be a non-empty string")

    if not re.match(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$", token):
        log.warning("Token suspicious format")
        raise ValueError("Token format is invalid - expected JWT format")

    try:
        payload = jwt.decode(token, _jwt_secret(), algorithm=_jwt_algorithm())
        user_id: str | None = payload.get("sub")
        if user_id is None:
            log.warning("Token missing sub")
            return None

        email = payload.get("email")
        if email is None:
            log.warning("Token missing email")
            return None

        log.info("Token verified", extra={"user_id": user_id, "email": email})
        return user_id, email

    except PyJWTError as e:
        log.error("Token verification failed", extra={"error": str(e)})
        return None
