from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

log = get_logger(__name__)


class DocMindError(Exception):
    """Base class for all domain errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class NotFoundError(DocMindError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(DocMindError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"
