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


def _toc_item_indices(doc: DoclingDocument) -> set[int]:
    """Indices of text items that contain a dot-leader + page number.

    Docling emits one text item per ToC line, so a single match marks
    an item as ToC. (The old `>= 2` threshold assumed multiple entries
    per item — that never happens with Docling output.)
    """
    return {idx for idx, item in enumerate(doc.texts) if _TOC_LINE.search(item.text or "")}


def _is_toc_document(doc: DoclingDocument, toc_indices: set[int]) -> bool:
    """5+ ToC-marked items, clustered in reading order, with ascending page numbers."""
    if len(toc_indices) < 5:
        return False

    # Cluster: largest run of ToC items within a 5-index window.
    indices = sorted(toc_indices)
    best_run = current = 1
    for a, b in zip(indices, indices[1:], strict=False):
        current = current + 1 if b - a <= 5 else 1
        best_run = max(best_run, current)

    if best_run < 5:
        return False

    # Monotonic: page numbers should mostly increase.
    pages: list[int] = []
    for idx in indices:
        for m in _TOC_LINE.finditer(doc.texts[idx].text or ""):
            pages.append(int(m.group(1)))
    ascending = sum(1 for a, b in zip(pages, pages[1:], strict=False) if b >= a)
    return ascending / (len(pages) - 1) > 0.8


def clean_document(doc: DoclingDocument) -> dict:
    """Normalize and prune in one pass. Returns stats."""
    changed = 0
    dropped_empty = 0

    # --- Phase 1: normalize every text item ---
    for item in doc.texts:
        original = item.text or ""
        cleaned = _clean_text(original)
        if cleaned != original:
            item.text = cleaned
            changed += 1
        if not cleaned:
            dropped_empty += 1

    toc_indices = _toc_item_indices(doc)
    toc_detected = _is_toc_document(doc, toc_indices)

    pruned = 0
    if toc_detected:
        before = len(doc.texts)
        doc.texts = [item for i, item in enumerate(doc.texts) if i not in toc_indices]
        pruned = before - len(doc.texts)
    else:
        log.info("prune_skipped", reason="no_toc_detected")

    stats = {
        "text_items": len(doc.texts),
        "changed": changed,
        "empty": dropped_empty,
        "toc_detected": toc_detected,
        "toc_items_found": len(toc_indices),
        "pruned": pruned,
    }
    log.info("cleaner_done", **stats)
    return stats
