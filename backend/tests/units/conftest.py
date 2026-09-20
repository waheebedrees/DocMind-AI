from unittest.mock import AsyncMock, MagicMock

import pytest
from app.db.repositories.user_repo import UserRepository


@pytest.fixture
def mock_user_repository():
    return MagicMock(spec=UserRepository)


@pytest.fixture
def pdf_bytes() -> bytes:
    # Minimal valid PDF — libmagic detects it as application/pdf.
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n" + b"x" * 2048


@pytest.fixture
def zip_bytes() -> bytes:
    # ZIP magic bytes — libmagic detects it, but it is not in ALLOWED_MIMES.
    return b"PK\x03\x04" + b"\x00" * 512
