"""Dump all chunks for a document (id, page, section, preview) so you can
pick chunk IDs for the eval set.

Usage:
    POSTGRES_HOST=localhost python scripts/dump_chunks.py \
        --document-id 7aa86487-54d2-472f-9c37-5a896dfb1ccf \
        --page 54
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db.session import AsyncSessionLocal
from app.models import DocumentChunk


async def main(args: argparse.Namespace) -> None:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.chunk_index,
                DocumentChunk.page_number,
                DocumentChunk.section,
                DocumentChunk.text,
            )
            .where(DocumentChunk.document_id == args.document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        if args.page is not None:
            stmt = stmt.where(DocumentChunk.page_number == args.page)

        rows = (await session.execute(stmt)).all()

    for r in rows:
        preview = r.text[:180].replace("\n", " ")
        print(f"[p{r.page_number} idx={r.chunk_index}] {r.id}")
        print(f"   section: {r.section or '—'}")
        print(f"   {preview}…")
        print()

    print(f"{len(rows)} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--page", type=int, default=None)
    asyncio.run(main(parser.parse_args()))
