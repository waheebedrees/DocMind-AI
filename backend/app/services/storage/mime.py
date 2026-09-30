"""Magic-byte MIME detection and extension mapping.

Layered detection strategy — each layer only runs when the previous
one returns an inconclusive result:

  1. Explicit binary signature table.
     Deterministic, dependency-free, and precise on truncated
     fixtures that libmagic refuses to commit to (PNG header + zeros,
     ELF header + zeros, etc.).

  2. OOXML marker scan inside ZIP containers.
     DOCX/XLSX/PPTX all share the ZIP magic `PK\\x03\\x04`; we look
     for their characteristic entry paths (`word/`, `xl/`, `ppt/`).

  3. Text sniffing (HTML / XML heuristics) on verified-text bytes.
     Crucially gated behind `_is_probably_text`, so a binary payload
     renamed to `.md` cannot be promoted to `text/markdown`.

  4. libmagic for the long tail.

  5. Extension fallback, restricted to verified text content.

Raises `UnsupportedMime` for anything outside the caller's allowlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import magic

from app.services.storage.base import UnsupportedMime

log = logging.getLogger(__name__)


ALLOWED_MIMES: Final[frozenset[str]] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",
        "text/markdown",
        "text/html",
        "image/png",
        "image/jpeg",
        "image/tiff",
    }
)

_MIME_TO_EXT: Final[dict[str, str]] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/html": ".html",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
}

# Non-canonical strings normalized to the canonical form expected by
# ALLOWED_MIMES and persisted in the database.
_MIME_ALIASES: Final[dict[str, str]] = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "application/x-pdf": "application/pdf",
    "text/x-markdown": "text/markdown",
    "text/x-c": "text/plain",
    "application/x-empty": "text/plain",
    "text/xml": "text/html",  # app treats XML payloads as HTML
}

# Extension -> MIME for text-format promotion. Only consulted after the
# byte stream has passed `_is_probably_text`.
_TEXT_EXT_TO_MIME: Final[dict[str, str]] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdown": "text/markdown",
    ".mkd": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
    ".log": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "text/html",
}

# MIME strings libmagic emits when it can't commit to a real format.
_GENERIC_MIMES: Final[frozenset[str]] = frozenset({"application/octet-stream", "text/plain"})


@dataclass(frozen=True, slots=True)
class _Signature:
    magic: bytes
    mime: str
    offset: int = 0
    label: str = ""


_BINARY_SIGNATURES: Final[tuple[_Signature, ...]] = (
    # Allowed types
    _Signature(b"\x89PNG\r\n\x1a\n", "image/png", 0, "PNG"),
    _Signature(b"\xff\xd8\xff", "image/jpeg", 0, "JPEG"),
    _Signature(b"%PDF-", "application/pdf", 0, "PDF"),
    _Signature(b"II*\x00", "image/tiff", 0, "TIFF-LE"),
    _Signature(b"MM\x00*", "image/tiff", 0, "TIFF-BE"),
    # Non-allowed types — declared so the reject path fires on the
    # first chunk with a precise reason, rather than falling through
    # to libmagic on a truncated fixture.
    _Signature(b"\x7fELF", "application/x-elf", 0, "ELF"),
    _Signature(b"MZ", "application/x-dosexec", 0, "PE"),
    _Signature(b"\xca\xfe\xba\xbe", "application/x-mach-binary", 0, "Mach-O"),
    _Signature(b"Rar!\x1a\x07\x00", "application/x-rar", 0, "RAR v4"),
    _Signature(b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed", 0, "7z"),
    _Signature(b"\x1f\x8b", "application/gzip", 0, "GZIP"),
    _Signature(b"BZh", "application/x-bzip2", 0, "BZIP2"),
)


_ZIP_MAGIC: Final[bytes] = b"PK\x03\x04"

_OOXML_MARKERS: Final[tuple[tuple[bytes, str], ...]] = (
    (
        b"word/",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    (
        b"xl/",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    (
        b"ppt/",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
)


_HTML_PREFIXES: Final[tuple[bytes, ...]] = (
    b"<!doctype html",
    b"<html",
    b"<head",
    b"<body",
)
_XML_PREFIXES: Final[tuple[bytes, ...]] = (b"<?xml",)

# Tuning knobs for `_is_probably_text`.
_TEXT_SNIFF_WINDOW: Final[int] = 4 * 1024  # bytes examined
_TEXT_CONTROL_BUDGET: Final[float] = 0.05  # max fraction of control chars


def _is_probably_text(data: bytes) -> bool:
    """Return True only if `data` looks like decodable, control-free text.

    This gates the extension-based fallback: a binary payload renamed
    to `.md` must not be promoted to `text/markdown` on the strength
    of its filename alone.
    """
    if not data:
        return False

    window = data[:_TEXT_SNIFF_WINDOW]

    # NUL bytes are the strongest single signal for "binary" and never
    # appear in legitimate UTF-8 text.
    if b"\x00" in window:
        return False

    try:
        text = window.decode("utf-8")
    except UnicodeDecodeError:
        return False

    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r\f")
    return controls / max(len(text), 1) < _TEXT_CONTROL_BUDGET


def _match_binary_signature(head: bytes) -> str | None:
    for sig in _BINARY_SIGNATURES:
        end = sig.offset + len(sig.magic)
        if len(head) >= end and head[sig.offset : end] == sig.magic:
            log.debug("mime.layer1 hit=%s mime=%s", sig.label, sig.mime)
            return sig.mime
    return None


def _match_ooxml(head: bytes) -> str | None:
    if not head.startswith(_ZIP_MAGIC):
        return None
    for marker, mime in _OOXML_MARKERS:
        if marker in head:
            log.debug("mime.layer2 marker=%r mime=%s", marker, mime)
            return mime
    return None


def _sniff_text(head: bytes, filename: str) -> str | None:
    """Return a text-family MIME if `head` looks like text, else None."""
    if not _is_probably_text(head):
        return None

    lowered = head[:256].lstrip().lower()
    if any(lowered.startswith(p) for p in _HTML_PREFIXES):
        log.debug("mime.layer3 html-prefix")
        return "text/html"
    if any(lowered.startswith(p) for p in _XML_PREFIXES):
        log.debug("mime.layer3 xml-prefix -> text/html")
        return "text/html"

    ext = Path(filename).suffix.lower()
    promoted = _TEXT_EXT_TO_MIME.get(ext)
    if promoted is None:
        log.debug("mime.layer3 plain-text ext=%r", ext)
        return "text/plain"

    log.debug("mime.layer3 extension-promoted ext=%r mime=%s", ext, promoted)
    return promoted


def _libmagic_lookup(head: bytes, filename: str) -> str:
    """Layer 4 + 5: libmagic, with a guarded extension fallback."""
    raw = magic.from_buffer(head, mime=True)
    mime = _MIME_ALIASES.get(raw, raw)

    if mime in _GENERIC_MIMES:
        if _is_probably_text(head):
            ext = Path(filename).suffix.lower()
            promoted = _TEXT_EXT_TO_MIME.get(ext)
            if promoted is not None:
                log.debug(
                    "mime.layer5 libmagic=%r ext=%r promoted=%s",
                    raw,
                    ext,
                    promoted,
                )
                return promoted
        log.debug("mime.layer4 libmagic=%r (no promotion)", raw)

    return mime


def _detect_raw(head: bytes, filename: str) -> str:
    # 1) Explicit binary signature.
    sig_mime = _match_binary_signature(head)
    if sig_mime is not None:
        return _MIME_ALIASES.get(sig_mime, sig_mime)

    # 2) OOXML inside a ZIP container.
    if head.startswith(_ZIP_MAGIC):
        ooxml = _match_ooxml(head)
        if ooxml is not None:
            return ooxml
        # Plain ZIP or OOXML whose markers fall past the head window:
        # hand off to libmagic, which reads the central directory.
        return _libmagic_lookup(head, filename)

    # 3) Text sniffing on verified-text bytes.
    text_mime = _sniff_text(head, filename)
    if text_mime is not None:
        return text_mime

    # 4) Long tail.
    return _libmagic_lookup(head, filename)


def detect_mime(
    head: bytes,
    filename: str,
    allowed: frozenset[str] = ALLOWED_MIMES,
) -> str:
    """Determine the MIME type from magic bytes.

    `head` must be the first N bytes of the file (N >= 1 KiB).
    `filename` is only consulted as a fallback for formats without
    reliable magic bytes *and* whose bytes have been verified as text.
    `allowed` is the acceptance set; the caller owns the policy.

    Raises:
        UnsupportedMime: if the resolved type is not in `allowed`.
    """
    if not head:
        raise UnsupportedMime("Empty file")

    mime = _detect_raw(head, filename)

    if mime not in allowed:
        raise UnsupportedMime(f"Detected MIME '{mime}' is not accepted. Allowed: {sorted(allowed)}")
    return mime


def extension_for(mime: str) -> str:
    """Canonical extension for a MIME type. Falls back to `.bin`."""
    return _MIME_TO_EXT.get(mime, ".bin")


__all__ = ["ALLOWED_MIMES", "detect_mime", "extension_for"]
