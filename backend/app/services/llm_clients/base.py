from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Provider-agnostic embedding interface.

    Contract:
      * Returns one vector per input text, in order.
      * Vector length equals `dimension`.
      * Vectors are normalized (cosine-similarity ready).
      * Raises PermanentEmbeddingError if the input can never succeed.
      * Raises TransientEmbeddingError if a retry might succeed.
      * Raises EmbeddingUnavailable if the backend itself is unusable.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
