from collections import defaultdict
from uuid import UUID

from app.rag.retrieval.types import Candidate, Passage


def build_passages(
    selected: list[Candidate],
    neighbors: list[Candidate],
    *,
    window: int,
    max_tokens: int,
) -> list[Passage]:
    """
    Merge each selected chunk with its ±window neighbors into contiguous passages.
    Passages are ordered by the best score inside them and filled until the token budget runs out.
    """

    by_pos: dict[tuple[UUID, int], Candidate] = {(c.document_id, c.chunk_index): c for c in neighbors}
    # selected rows carry scores
    by_pos.update({(c.document_id, c.chunk_index): c for c in selected})

    # chunk positions to include, per document
    wanted: dict[UUID, dict[int, float]] = defaultdict(dict)
    for c in selected:
        for i in range(c.chunk_index - window, c.chunk_index + window + 1):
            if (c.document_id, i) in by_pos:
                prev = wanted[c.document_id].get(i, float("-inf"))
                wanted[c.document_id][i] = max(prev, c.relevance)

    # group consecutive indices into runs
    runs: list[tuple[float, list[Candidate]]] = []
    for doc_id, idx_scores in wanted.items():
        run: list[int] = []
        for i in sorted(idx_scores):
            if run and i != run[-1] + 1:
                runs.append((max(idx_scores[j] for j in run), [by_pos[(doc_id, j)] for j in run]))
                run = []
            run.append(i)
        if run:
            runs.append((max(idx_scores[j] for j in run), [by_pos[(doc_id, j)] for j in run]))

    runs.sort(key=lambda r: r[0], reverse=True)

    passages: list[Passage] = []
    used = 0
    for score, chunks in runs:
        cost = sum(c.tokens for c in chunks)
        if used + cost > max_tokens:
            if passages:
                continue  # a smaller later passage may still fit
            chunks = chunks[:1]  # always return at least the best chunk
            cost = chunks[0].tokens
        pages = [c.page_number for c in chunks if c.page_number is not None]
        passages.append(
            Passage(
                citation_id=len(passages) + 1,
                document_id=chunks[0].document_id,
                chunk_ids=tuple(c.chunk_id for c in chunks),
                chunk_range=(chunks[0].chunk_index, chunks[-1].chunk_index),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                section=next((c.section for c in chunks if c.section), None),
                text="\n".join(c.text for c in chunks),
                score=score,
                token_count=cost,
            )
        )
        used += cost
    return passages


def render_context(passages: list[Passage]) -> str:
    blocks = []
    for p in passages:
        loc = f"p.{p.page_start}" if p.page_start == p.page_end else f"pp.{p.page_start}-{p.page_end}"
        header = f"[{p.citation_id}] {p.section or ''} ({loc if p.page_start else 'n/a'})".strip()
        blocks.append(f"{header}\n{p.text}")
    return "\n\n---\n\n".join(blocks)
