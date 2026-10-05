from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID


@dataclass
class EvalCase:
    """One query with its ground-truth annotations."""

    id: str
    query: str
    user_id: UUID
    document_ids: list[UUID]
    # chunk IDs that MUST appear for this query to be "answered"
    relevant_chunk_ids: set[str]
    answerable: bool = True
    reference_answer: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> EvalCase:
        return cls(
            id=d["id"],
            query=d["query"],
            user_id=UUID(d["user_id"]),
            document_ids=[UUID(x) for x in d.get("document_ids", [])],
            relevant_chunk_ids=set(d.get("relevant_chunk_ids", [])),
            answerable=d.get("answerable", True),
            reference_answer=d.get("reference_answer", ""),
            tags=d.get("tags", []),
        )


@dataclass
class EvalResult:
    """Result of running one eval case through the pipeline."""

    case: EvalCase
    # chunk IDs returned at each stage, ordered by rank
    stage_chunks: dict[str, list[str]]
    # raw scores at each stage
    stage_scores: dict[str, list[float]]
    # computed metrics per stage
    metrics: dict[str, dict[str, float | None]]
    # timing
    timings_ms: dict[str, float]
    reranked: bool
    # final passages text (for answer eval)
    passages_text: list[str]
    total_ms: float = 0.0


def load_eval_set(path: str | Path) -> list[EvalCase]:
    """Load JSONL eval set."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def save_eval_set(cases: list[EvalCase], path: str | Path) -> None:
    """Save eval set as JSONL."""
    with open(path, "w") as f:
        for c in cases:
            d = {
                "id": c.id,
                "query": c.query,
                "user_id": str(c.user_id),
                "document_ids": [str(x) for x in c.document_ids],
                "relevant_chunk_ids": sorted(c.relevant_chunk_ids),
                "answerable": c.answerable,
                "reference_answer": c.reference_answer,
                "tags": c.tags,
            }
            f.write(json.dumps(d) + "\n")
