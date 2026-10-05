"""Normalize text in a DoclingDocument before chunking.

Two responsibilities, in order:

1. Text normalization — unicode NFKC, zero-width stripping, whitespace
   collapse. Applies to every text item.
2. Non-content pruning — drop ToC dot-leader items. Gated on document-level
   ToC detection, so contracts, code, decimal tables, and IP lists pass
   through untouched.

Mutates doc in place and returns stats.
"""

import re
import unicodedata

from docling_core.types.doc import DoclingDocument

from app.core.logging import get_logger

log = get_logger(__name__)

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Dot-leader followed by a page number. No `$` anchor: Docling joins
# multiple ToC entries onto one line, so an end-of-line anchor only
# matches the last entry per item. Capture the page number directly
# rather than re-scanning with re.search(r"\d+"), which would grab a
# digit from the title portion (e.g. "chapter 4....264" -> 4).
_TOC_LINE = re.compile(r"\.{3,}\s*(\d+)")


def _clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH.sub("", s)
    s = _MULTI_SPACE.sub(" ", s)
    s = _MULTI_NEWLINE.sub("\n\n", s)
    return s.strip()


def _toc_items(doc: DoclingDocument) -> list:
    """Text items that contain a dot-leader + page number."""
    return [item for item in doc.texts if _TOC_LINE.search(item.text or "")]


def _is_toc_document(doc: DoclingDocument, toc_items: list) -> bool:
    """5+ ToC-marked items, clustered in reading order, with ascending page numbers."""
    if len(toc_items) < 5:
        return False

    # Reading-order positions come from each item's self_ref (`#/texts/N`).
    indices: list[int] = []
    for item in toc_items:
        ref = getattr(item, "self_ref", "")
        try:
            indices.append(int(ref.rsplit("/", 1)[1]))
        except (ValueError, IndexError):
            indices.append(doc.texts.index(item))
    indices.sort()

    best_run = current = 1
    for a, b in zip(indices, indices[1:], strict=False):
        current = current + 1 if b - a <= 5 else 1
        best_run = max(best_run, current)
    if best_run < 5:
        return False

    pages: list[int] = []
    for item in toc_items:
        for m in _TOC_LINE.finditer(item.text or ""):
            pages.append(int(m.group(1)))

    if len(pages) < 2:  # guard the divisor
        return False
    ascending = sum(1 for a, b in zip(pages, pages[1:], strict=False) if b >= a)
    return ascending / (len(pages) - 1) > 0.8


def clean_document(doc: DoclingDocument) -> dict:
    """Normalize and prune in one pass. Returns stats."""
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

    toc_items = _toc_items(doc)
    toc_detected = _is_toc_document(doc, toc_items)

    pruned = 0
    if toc_detected:
        pruned = len(toc_items)
        # delete_items updates self_refs, parent pointers, group children,
        # floating-item captions/references/footnotes, and rich-cell refs,
        # so the ref graph survives model_validate_json on reload. The old
        # list-filter approach shifted indices without renumbering refs,
        # which is what tripped the hierarchy validator.
        doc.delete_items(node_items=toc_items)
    else:
        log.info("prune_skipped", reason="no_toc_detected")

    stats = {
        "text_items": len(doc.texts),
        "changed": changed,
        "empty": dropped_empty,
        "toc_detected": toc_detected,
        "toc_items_found": len(toc_items),
        "pruned": pruned,
    }
    log.info("cleaner_done", **stats)
    return stats
