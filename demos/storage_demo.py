
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

load_dotenv("./envs/dev.env")


import asyncio
from uuid import uuid4

from backend.app.services.storage import (
    ObjectNotFound,
    LocalStorage,
    UnsupportedMime,
    InvalidKey,
    UploadTooLarge
)


# A tiny valid PDF header. libmagic only needs the first few bytes
# to recognize "%PDF-", so this is enough for detection even though
# it isn't a real PDF.
FAKE_PDF = b"%PDF-1.4\n" + b"x" * 4096
MARKDOWN = b"# Title\n\nSome *markdown* text.\n"
HTML = b"<!doctype html>\n<html><body>hi</body></html>\n"
PLAIN = b"just some notes\n"
JUNK = b"\x00\x01\x02\x03 not a known type \xff\xfe"

# --------------------------------------------------------------------


async def stream_from_bytes(data: bytes, chunk: int = 64 * 1024):
    for i in range(0, len(data), chunk):
        yield data[i: i + chunk]


async def stream_file(path: Path, chunk: int = 64 * 1024):
    """Yield a real file's bytes in chunks — same shape as iter_upload."""
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            yield data
            

async def main() -> None:
    root = Path("./storage_demo")
    storage = LocalStorage(root)

    # 1. Startup — creates {root} and {root}/tmp. Idempotent.
    await storage.startup()
    print(f"root ready: {root.resolve()}\n")

    user = uuid4()
  # ----------------------------------------------------------------
    # 5b. Real .docx from disk — exercises the full pipeline against
    #     a real OOXML file (2.6 MB, larger than the 8 KB head buffer,
    #     so the early-sniff path fires).
    # ----------------------------------------------------------------
    print("== upload: real .docx ==")
    docx_path = Path("./docs/ZeroStrike3.docx")

    if not docx_path.exists():
        print(f"  skipped — not found: {docx_path.resolve()}\n")
    else:
        raw_docx = docx_path.read_bytes()
        docx_obj = await storage.put_stream(
            user_id=user,
            stream=stream_file(docx_path),
            filename=docx_path.name,
            max_bytes=50 * 1024 * 1024,
        )
        print(f"  key        = {docx_obj.key}")
        print(f"  mime       = {docx_obj.mime_type}")
        print(f"  size       = {docx_obj.size_bytes}")
        print(f"  dedup      = {docx_obj.deduplicated}")

        # round-trip check
        collected = bytearray()
        async for chunk in storage.get(docx_obj.key):
            collected.extend(chunk)
            
        print(f"  identical  = {bytes(collected) == raw_docx}\n")
        
    # ----------------------------------------------------------------
    # 2. Upload a PDF — MIME sniffed from magic bytes.
    # ----------------------------------------------------------------
    print("== upload: PDF ==")
    pdf_obj = await storage.put_stream(
        user_id=user,
        stream=stream_from_bytes(FAKE_PDF),
        filename="report.pdf",
        max_bytes=1_000_000,
    )
    print(f"  key        = {pdf_obj.key}")
    print(f"  mime       = {pdf_obj.mime_type}")
    print(f"  size       = {pdf_obj.size_bytes}")
    print(f"  dedup      = {pdf_obj.deduplicated}")
    print(f"  hash       = {pdf_obj.content_hash[:16]}…\n")

    # ----------------------------------------------------------------
    # 3. Upload markdown. libmagic reports text/plain, so the
    #    extension fallback in detect_mime() promotes it to
    #    text/markdown.
    # ----------------------------------------------------------------
    print("== upload: markdown ==")
    md_obj = await storage.put_stream(
        user_id=user,
        stream=stream_from_bytes(MARKDOWN),
        filename="notes.md",
        max_bytes=1_000_000,
    )
    print(f"  mime       = {md_obj.mime_type}   (expected text/markdown)")
    print(f"  key        = {md_obj.key}\n")

    # ----------------------------------------------------------------
    # 4. Upload HTML. libmagic is unreliable here, but the
    #    _HTML_PREFIXES check catches the doctype.
    # ----------------------------------------------------------------
    print("== upload: html ==")
    html_obj = await storage.put_stream(
        user_id=user,
        stream=stream_from_bytes(HTML),
        filename="page.html",
        max_bytes=1_000_000,
    )
    
    
    print(f"  mime       = {html_obj.mime_type}\n")

    # ----------------------------------------------------------------
    # 5. Deduplication. Same bytes, same user → same key,
    #    deduplicated=True.
    # ----------------------------------------------------------------
    print("== upload: identical PDF again ==")
    dup = await storage.put_stream(
        user_id=user,
        stream=stream_from_bytes(FAKE_PDF),
        filename="report-copy.pdf",
        max_bytes=1_000_000,
    )
    print(f"  same key   = {dup.key == pdf_obj.key}")
    print(f"  dedup      = {dup.deduplicated}\n")

    # ----------------------------------------------------------------
    # 6. Rejections: bad type, oversize, empty.
    # ----------------------------------------------------------------
    print("== rejections ==")

    try:
        await storage.put_stream(
            user_id=user,
            stream=stream_from_bytes(JUNK),
            filename="blob.bin",
            max_bytes=1_000_000,
        )
    except UnsupportedMime as exc:
        print(f"  UnsupportedMime: {exc}")

    try:
        await storage.put_stream(
            user_id=user,
            stream=stream_from_bytes(PLAIN),
            filename="big.txt",
            max_bytes=10,       # cap smaller than the payload
        )
    except UploadTooLarge as exc:
        print(f"  UploadTooLarge: {exc}")

    try:
        await storage.put_stream(
            user_id=user,
            stream=stream_from_bytes(b""),
            filename="empty.txt",
            max_bytes=1_000_000,
        )
    except UnsupportedMime as exc:
        print(f"  UnsupportedMime: {exc}")
    print()

    # ----------------------------------------------------------------
    # 7. exists() and reading back.
    # ----------------------------------------------------------------
    print("== read back ==")
    print(f"  exists(pdf)         = {await storage.exists(pdf_obj.key)}")

    bogus = storage.key_for(
        user_id=user, content_hash="b" * 64, extension=".pdf",
    )
    print(f"  exists(bogus)       = {await storage.exists(bogus)}")

    collected = bytearray()
    async for chunk in storage.get(pdf_obj.key):
        collected.extend(chunk)
    print(f"  bytes read          = {len(collected)}")
    print(f"  round-trips exactly = {bytes(collected) == FAKE_PDF}\n")

    try:
        await storage.exists("nope/..")
    except InvalidKey as exc:
        print(f"  InvalidKey: {exc}")
        
    # ----------------------------------------------------------------
    # 8. open_stream() vs get() on a missing key.
    # ----------------------------------------------------------------
    print("== missing key ==")
    missing = storage.key_for(
        user_id=user, content_hash="a" * 64, extension=".pdf",
    )
    try:
        await storage.open_stream(missing)
    except ObjectNotFound as exc:
        print(f"  open_stream: ObjectNotFound ({exc})")

    # get() defers the failure until the first iteration.
    try:
        async for _ in storage.get(missing):
            pass
    except ObjectNotFound as exc:
        print(f"  get (iter):  ObjectNotFound ({exc})\n")


    # ----------------------------------------------------------------
    # 9. materialize() — a real path for external libraries.
    # ----------------------------------------------------------------
    print("== materialize ==")
    async with storage.materialize(pdf_obj.key) as path:
        print(f"  path        = {path}")
        print(f"  is_file     = {path.is_file()}")
        print(f"  first 8     = {path.read_bytes()[:8]!r}\n")


    # ----------------------------------------------------------------
    # 10. delete() — idempotent.
    # ----------------------------------------------------------------
    print("== delete ==")
    await storage.delete(pdf_obj.key)
    print(f"  after delete, exists = {await storage.exists(pdf_obj.key)}")
    await storage.delete(pdf_obj.key)   # second call: no error
    print("  second delete: ok (idempotent)")

    
if __name__ == "__main__":
    asyncio.run(main())
