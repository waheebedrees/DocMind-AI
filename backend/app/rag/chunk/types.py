from dataclasses import dataclass

from app.core.exceptions import DocMindError


class ChunkingError(DocMindError):
    """Raised when a chunk exceeds the embedding model limit unexpectedly."""


@dataclass(frozen=True)
class PreparedChunk:
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None
    doc_item_labels: tuple[str, ...]
    section: str | None
    embedding: list[float] | None
