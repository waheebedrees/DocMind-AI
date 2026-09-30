import pytest
from app.services.storage.base import UnsupportedMime
from app.services.storage.mime import detect_mime, extension_for

PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 64
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000") + b"\x00" * 64

TIFF_LE = b"II*\x00" + b"\x00" * 64
TIFF_BE = b"MM\x00*" + b"\x00" * 64
ELF = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64
PE = b"MZ\x90\x00" + b"\x00" * 64


def test_png_by_magic():
    assert detect_mime(PNG, "whatever.txt") == "image/png"


def test_pdf_by_magic():
    assert detect_mime(PDF, "report.pdf") == "application/pdf"


def test_jpeg_by_magic():
    assert detect_mime(JPEG, "photo.jpg") == "image/jpeg"


def test_magic_beats_extension():
    # A PNG named .txt is still a PNG.
    assert detect_mime(PNG, "sneaky.txt") == "image/png"


def test_markdown_promoted_from_text_plain():
    """Regression: libmagic returns text/plain for markdown, and the
    old code only consulted the extension for octet-stream."""
    head = b"# Heading\n\nSome *markdown* body text.\n"
    assert detect_mime(head, "notes.md") == "text/markdown"
    assert detect_mime(head, "notes.markdown") == "text/markdown"


def test_plain_text_stays_plain():
    head = b"just some words in a file\n"
    assert detect_mime(head, "notes.txt") == "text/plain"


def test_unknown_extension_stays_plain():
    head = b"just some words in a file\n"
    assert detect_mime(head, "notes.log") == "text/plain"


@pytest.mark.parametrize(
    "head",
    [
        b"<!DOCTYPE html>\n<html><body>hi</body></html>",
        b"  \n<html><head></head></html>",
        b"<?xml version='1.0'?><root/>",
    ],
)
def test_html_sniffed_from_prefix(head):
    assert detect_mime(head, "page.bin") == "text/html"


def test_empty_head_rejected():
    with pytest.raises(UnsupportedMime, match="Empty file"):
        detect_mime(b"", "empty.txt")


def test_disallowed_mime_rejected():
    elf = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64
    with pytest.raises(UnsupportedMime):
        detect_mime(elf, "binary")


def test_allowed_set_is_a_parameter():
    """Callers own the policy; detect_mime must not hardcode it."""
    assert detect_mime(PNG, "a.png", frozenset({"image/png"})) == "image/png"
    with pytest.raises(UnsupportedMime):
        detect_mime(PNG, "a.png", frozenset({"application/pdf"}))


def test_extension_for_known_and_unknown():
    assert extension_for("image/png") == ".png"
    assert extension_for("text/markdown") == ".md"
    assert extension_for("application/x-nonsense") == ".bin"


def test_tiff_both_endiannesses():
    assert detect_mime(TIFF_LE, "scan.bin") == "image/tiff"
    assert detect_mime(TIFF_BE, "scan.bin") == "image/tiff"


def test_binary_payload_renamed_as_markdown_is_rejected():
    """Regression for the extension-fallback security hole."""
    with pytest.raises(UnsupportedMime):
        detect_mime(ELF, "notes.md")


def test_pe_executable_rejected_early():
    with pytest.raises(UnsupportedMime):
        detect_mime(PE, "invoice.pdf")


def test_tiff_by_magic():
    assert detect_mime(TIFF_LE, "scan.bin") == "image/tiff"
    assert detect_mime(TIFF_BE, "scan.bin") == "image/tiff"
