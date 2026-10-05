"""Find chunk IDs for eval set annotation.

Usage:
    python scripts/annotate_chunks.py \
        --document-id 7aa86487-54d2-472f-9c37-5a896dfb1ccf \
        --search "nine core tables"
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
        stmt = select(
            DocumentChunk.id,
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            DocumentChunk.section,
            DocumentChunk.text,
        ).where(DocumentChunk.document_id == args.document_id)

        if args.search:
            stmt = stmt.where(DocumentChunk.text.ilike(f"%{args.search}%"))

        if args.page:
            stmt = stmt.where(DocumentChunk.page_number == args.page)

        stmt = stmt.order_by(DocumentChunk.chunk_index)

        rows = (await session.execute(stmt)).all()

        for r in rows:
            preview = r.text[:150].replace("\n", " ")
            print(f"chunk_id={r.id}  idx={r.chunk_index}  page={r.page_number}  section={r.section or '—'}")
            print(f"  {preview}...")
            print()

        print(f"\n{len(rows)} chunks found")


async def go():
    async with AsyncSessionLocal() as s:
        for cid in [
            "176e5301-c93b-4977-b182-10ab01a70785",  # p28 NFRs
            "9e02d8d5-a229-4fc9-8273-67fbd5239acd",  # p65 Achievements
            "0cc4a7b0-3ec3-483e-b06e-91f21127bcce",  # p11 Aim
        ]:
            r = (
                await s.execute(select(DocumentChunk.chunk_index, DocumentChunk.page_number, DocumentChunk.text).where(DocumentChunk.id == cid))
            ).one()
            print(f"=== p{r.page_number} idx={r.chunk_index} {cid} ===")
            print(r.text)
            print()
    async with AsyncSessionLocal() as s:
        for cid in [
            "40a2768a-759e-4777-85bf-a561bd5a6837",  # idx=111
            # idx=109 "Execution, Storage, and Intelligence Components"
            "0e75f017-bf99-4d40-9ace-5ac0c03b3c2f",
            "7b064e80-20d9-4bf5-bbcd-b3bcc5d7132f",  # idx=108 "Agent Architecture"
        ]:
            r = (await s.execute(select(DocumentChunk.id, DocumentChunk.chunk_index, DocumentChunk.text).where(DocumentChunk.id == cid))).one()
            print(f"=== idx={r.chunk_index} {r.id} ===")
            print(r.text)
            print()

    async with AsyncSessionLocal() as s:
        for cid in [
            "176e5301-c93b-4977-b182-10ab01a70785",  # p28 NFRs
            "9e02d8d5-a229-4fc9-8273-67fbd5239acd",  # p65 Achievements
            "0cc4a7b0-3ec3-483e-b06e-91f21127bcce",  # p11 Aim
        ]:
            r = (
                await s.execute(select(DocumentChunk.chunk_index, DocumentChunk.page_number, DocumentChunk.text).where(DocumentChunk.id == cid))
            ).one()
            print(f"=== p{r.page_number} idx={r.chunk_index} {cid} ===")
            print(r.text)
            print()


# asyncio.run(go())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--search", default=None)
    parser.add_argument("--page", type=int, default=None)
    asyncio.run(main(parser.parse_args()))
