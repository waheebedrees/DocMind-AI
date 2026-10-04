"""Centralized exception hierarchy.

Layering rules:
  * Every domain exception inherits from DocMindError (so the FastAPI
    exception handler can map it to a response).
  * PermanentError / TransientError are *retry-classification* markers.
    Anything that must never be retried inherits PermanentError; anything
    that should be retried inherits TransientError.
"""

from fastapi import status


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
    """HTTP-mapped errors"""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(DocMindError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class DocumentNotReady(DocMindError):
    status_code = status.HTTP_409_CONFLICT
    code = "document_not_ready"


# Retry-classification markers
class PermanentError(DocMindError):
    """Do NOT retry: bad input, missing artifact, config mismatch."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "permanent_error"


class TransientError(DocMindError):
    """Retry: rate limit, network blip, timeout, 5xx upstream."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "transient_error"


# Embedding-specific
class EmbeddingError(DocMindError):
    """Catch-all for any embedding-pipeline failure."""


class PermanentEmbeddingError(EmbeddingError, PermanentError):
    """This call can never succeed. Wrong dimension, model not found,
    bad input. Do not retry."""

    code = "permanent_embedding_error"


class TransientEmbeddingError(EmbeddingError, TransientError):
    """This call failed but might succeed on retry. Timeout, 429,
    connection reset, upstream 5xx."""

    code = "transient_embedding_error"


class EmbeddingUnavailable(EmbeddingError, PermanentError):
    """The embedding deployment is unusable. Operator must fix config
    or bring the backend up. Not retryable, but surface as 503 because
    the endpoint is temporarily (from the client's view) unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "embedding_unavailable"


class InvalidTransition(PermanentError):
    """A caller tried a state transition the state machine forbids.

    Permanent by design: re-running the same job from the same state
    will fail identically, so ``run_stage`` must not retry it.
    """


class JobNotFound(NotFoundError):
    """The job row does not exist. Likely the document was deleted."""


class DocumentNotFound(NotFoundError):
    """The document row does not exist but a job references it."""


class JobAlreadyDone(PermanentError):
    """Duplicate delivery: the job already completed successfully."""


class JobAlreadyFailed(PermanentError):
    """Duplicate delivery: the job already failed permanently."""
