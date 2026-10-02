"""Normalize text in a DoclingDocument before chunking.

Docling already handles layout, reading order, and table structure.
This pass only touches text content: unicode normalization, control
character removal, whitespace collapse. It mutates doc in place and
returns stats.
"""

import re
import unicodedata

from docling_core.types.doc import DoclingDocument

from app.core.logging import get_logger

log = get_logger(__name__)

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH.sub("", s)
    s = _MULTI_SPACE.sub(" ", s)
    s = _MULTI_NEWLINE.sub("\n\n", s)
    return s.strip()


def clean_document(doc: DoclingDocument) -> dict:
    """Mutate text items in place. Returns stats."""
    changed = 0
    dropped_empty = 0

    for item in doc.texts:
        original = item.text or ""
        cleaned = _clean_text(original)
        if cleaned != original:
            item.text = cleaned
            changed += 1
        if not cleaned:
            dropped_empty += 1

    stats = {
        "text_items": len(doc.texts),
        "changed": changed,
        "empty": dropped_empty,
    }
    log.info("cleaner_done", **stats)
    return stats
