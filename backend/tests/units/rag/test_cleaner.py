"""Tests for the document cleaner (text normalization + ToC pruning).

clean_document mutates a DoclingDocument in place and returns a stats
dict. We fake the document with a minimal object exposing only what the
cleaner touches: a mutable `texts` list whose items have a `.text`
attribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.rag.cleaning import (
    _clean_text,
    _is_toc_document,
    _toc_item_indices,
    clean_document,
)


@dataclass
class _Item:
    text: str | None


class _FakeDoc:
    def __init__(self, texts: list[str | None]):
        self.texts = [_Item(t) for t in texts]


# --- _clean_text ----------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # NFKC normalization
        ("\ufb01le", "file"),              # ﬁ ligature
        ("\uff11\uff12\uff13", "123"),     # fullwidth digits
        ("\uff21\uff22\uff23", "ABC"),     # fullwidth letters
        ("\u00a0hello", "hello"),          # NBSP -> space -> strip
        # zero-width stripping
        ("a\u200bb", "ab"),
        ("a\u200cb", "ab"),
        ("a\u200db", "ab"),
        ("a\ufeffb", "ab"),
        # whitespace collapse
        ("a  b", "a b"),
        ("a\t\tb", "a b"),
        ("a \t b", "a b"),
        # newline collapse
        ("a\n\n\nb", "a\n\nb"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("a\n\nb", "a\n\nb"),              # unchanged
        ("a\nb", "a\nb"),                  # unchanged
        # strip
        ("  hello  ", "hello"),
        ("\n\nhello\n\n", "hello"),
        # already clean
        ("hello world", "hello world"),
        ("hello, world.", "hello, world."),
        ("line one\nline two", "line one\nline two"),
        # empty
        ("", ""),
        ("   ", ""),
        ("\t\t\t", ""),
        ("\u200b\u200b", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert _clean_text(raw) == expected


# --- _toc_item_indices ----------------------------------------------


@pytest.mark.parametrize(
    "texts, expected",
    [
        ([], set()),
        (["no dots here"], set()),
        (["a........5"], {0}),
        (["a........5", "b........6"], {0, 1}),
        (["....5", "clean", "...10"], {0, 2}),
        ([None], set()),
        (["..5"], set()),             # only 2 dots
        (["....no digits"], set()),   # no page number
        (["item 1.2.3"], set()),      # single dots between digits
    ],
)
def test_toc_item_indices(texts, expected):
    doc = _FakeDoc(texts)
    assert _toc_item_indices(doc) == expected


# --- _is_toc_document -----------------------------------------------


def _toc_doc(pages: list[int]) -> _FakeDoc:
    """Build a doc of ToC-like lines with the given page numbers."""
    return _FakeDoc([f"Item {i}........{p}" for i, p in enumerate(pages)])


def test_is_toc_below_threshold():
    doc = _toc_doc([1, 2, 3, 4])
    assert _is_toc_document(doc, _toc_item_indices(doc)) is False


def test_is_toc_clustered_and_ascending():
    doc = _toc_doc([1, 11, 21, 31, 41])
    assert _is_toc_document(doc, _toc_item_indices(doc)) is True


def test_is_toc_unclustered_fails():
    texts = ["clean"] * 41
    for i in (0, 10, 20, 30, 40):
        texts[i] = f"Item........{i + 1}"
    doc = _FakeDoc(texts)
    assert _is_toc_document(doc, _toc_item_indices(doc)) is False


def test_is_toc_not_ascending_fails():
    doc = _toc_doc([50, 40, 30, 20, 10])
    assert _is_toc_document(doc, _toc_item_indices(doc)) is False


def test_is_toc_empty_document():
    doc = _FakeDoc([])
    assert _is_toc_document(doc, _toc_item_indices(doc)) is False


# --- clean_document -------------------------------------------------


def test_clean_document_normalizes_and_counts():
    doc = _FakeDoc(["  hello  ", "a\u200bb", "multi   space"])
    stats = clean_document(doc)
    assert [t.text for t in doc.texts] == ["hello", "ab", "multi space"]
    assert stats["changed"] == 3
    assert stats["empty"] == 0


def test_clean_document_counts_empty_items():
    doc = _FakeDoc(["hi", "   ", None, "\u200b"])
    stats = clean_document(doc)
    # whitespace-only, None, and zero-width all count as empty
    assert stats["empty"] == 3
    # "   " -> "" and zero-width -> "" count as changed; None and "hi" do not
    assert stats["changed"] == 2


def test_clean_document_returns_expected_stat_keys():
    stats = clean_document(_FakeDoc(["hi"]))
    assert set(stats.keys()) == {
        "text_items",
        "changed",
        "empty",
        "toc_detected",
        "toc_items_found",
        "pruned",
    }


def test_clean_document_empty_document():
    stats = clean_document(_FakeDoc([]))
    assert stats["text_items"] == 0
    assert stats["changed"] == 0
    assert stats["toc_detected"] is False


def test_clean_document_prunes_detected_toc():
    toc = [
        "Introduction........1",
        "Chapter One.........5",
        "Chapter Two.........12",
        "Chapter Three.......20",
        "Chapter Four........30",
    ]
    body = ["body one", "body two", "body three"]
    doc = _FakeDoc(toc + body)
    stats = clean_document(doc)

    assert stats["toc_detected"] is True
    assert stats["toc_items_found"] == 5
    assert stats["pruned"] == 5
    assert stats["text_items"] == 3
    assert [t.text for t in doc.texts] == body


def test_clean_document_does_not_prune_when_no_toc():
    doc = _FakeDoc(
        [
            "Article 1.2.3 of the contract",
            "IP 192.168.1.1",
            "See section 4.5",
            "Plain text.",
        ]
    )
    stats = clean_document(doc)
    assert stats["toc_detected"] is False
    assert stats["pruned"] == 0
    assert stats["text_items"] == 4


def test_clean_document_does_not_prune_below_toc_threshold():
    doc = _FakeDoc(
        [
            "Alpha........1",
            "Beta.........2",
            "Gamma........3",
            "regular body text",
        ]
    )
    stats = clean_document(doc)
    assert stats["toc_detected"] is False
    assert stats["toc_items_found"] == 3
    assert stats["pruned"] == 0


def test_clean_runs_before_toc_detection():
    """Dot leaders broken by zero-width chars only become detectable
    after cleaning, so cleaning must run first."""
    zw = "\u200b"
    toc = [f"Item {i}{zw}.{zw}.{zw}.{zw}.{i * 10 + 1}" for i in range(5)]
    doc = _FakeDoc(toc)
    stats = clean_document(doc)
    assert stats["toc_detected"] is True
    assert stats["pruned"] == 5
