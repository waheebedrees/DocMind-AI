"""Build a draft eval_set.jsonl by searching each query's keywords.

Usage:
    uv run python scripts/build_eval_set.py \
        --document-id 7aa86487-54d2-472f-9c37-5a896dfb1ccf \
        --user-id d7eb5ff5-88bb-4c5b-b388-f7132ce8a736 \
        --queries scripts/eval_queries.json \
        --output tests/eval/fixtures/eval_set.draft.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiofiles
from sqlalchemy import or_, select

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db.session import AsyncSessionLocal
from app.models import DocumentChunk


async def find_chunks(session, document_id, keywords: list[str], limit: int = 5):
    """Return chunks whose text matches any keyword, ranked by match count."""
    stmt = select(
        DocumentChunk.id,
        DocumentChunk.chunk_index,
        DocumentChunk.page_number,
        DocumentChunk.section,
        DocumentChunk.text,
    ).where(DocumentChunk.document_id == document_id)

    if keywords:
        stmt = stmt.where(or_(*[DocumentChunk.text.ilike(f"%{kw}%") for kw in keywords]))

    stmt = stmt.order_by(DocumentChunk.chunk_index)
    rows = (await session.execute(stmt)).all()

    # Rank by how many keywords matched
    def score(row):
        text = (row.text or "").lower()
        return sum(1 for kw in keywords if kw.lower() in text)

    rows = sorted(rows, key=score, reverse=True)
    return rows[:limit]


async def main(args):

    queries = json.loads(await asyncio.to_thread(Path(args.queries).read_text))

    async with AsyncSessionLocal() as session:
        for q in queries:
            matches = await find_chunks(session, args.document_id, q["keywords"])
            q["_matches"] = [
                {
                    "chunk_id": str(r.id),
                    "chunk_index": r.chunk_index,
                    "page": r.page_number,
                    "section": r.section,
                    "preview": (r.text or "")[:140].replace("\n", " "),
                }
                for r in matches
            ]

    # Write the draft so you can inspect candidates
    draft_path = Path(args.output)
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(draft_path, "w") as f:
        for q in queries:
            print(f"\n=== {q['id']} ===")
            print(f"query: {q['query']}")
            print(f"keywords: {q['keywords']}")
            print("candidates:")
            for m in q["_matches"]:
                print(f"  {m['chunk_id']}  p.{m['page']}  {m['section'] or '—'}")
                print(f"    {m['preview']}")

            # Emit a draft JSONL line with the top candidate pre-filled.
            # You'll edit this to pick the right one(s).
            top = q["_matches"][0]["chunk_id"] if q["_matches"] else ""
            line = {
                "id": q["id"],
                "query": q["query"],
                "user_id": args.user_id,
                "document_ids": [args.document_id],
                "relevant_chunk_ids": [top] if top else [],
                "answerable": q.get("answerable", True),
                "reference_answer": q.get("reference_answer", ""),
                "tags": q.get("tags", []),
            }
            await f.write(json.dumps(line) + "\n")

    print(f"\nDraft written to {draft_path}")
    print("Review the candidates above and edit the JSONL before running eval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
