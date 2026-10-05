from dataclasses import dataclass, field
from uuid import UUID

import numpy as np


@dataclass(slots=True)
class Candidate:
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    page_number: int | None
    section: str | None
    token_count: int | None
    embedding: np.ndarray | None = None

    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def relevance(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score

    @property
    def tokens(self) -> int:
        return self.token_count or max(1, len(self.text) // 4)


@dataclass(frozen=True, slots=True)
class Passage:
    """One or more consecutive chunks from the same document, cited as a unit."""

    citation_id: int
    document_id: UUID
    chunk_ids: tuple[UUID, ...]
    chunk_range: tuple[int, int]
    page_start: int | None
    page_end: int | None
    section: str | None
    text: str
    score: float
    token_count: int


@dataclass(slots=True)
class RetrievalResult:
    query: str
    passages: list[Passage]
    context: str
    reranked: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    stages: dict[str, list[Candidate]] = field(default_factory=dict)
