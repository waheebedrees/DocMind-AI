import contextlib
import os
from pathlib import Path
from typing import BinaryIO


def flush_sync(fh: BinaryIO) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def rmdir_quiet(path: Path) -> None:
    # Path.rmdir() takes no arguments — no missing_ok. It fails on
    # non-empty and missing dirs, which is exactly what we want.
    with contextlib.suppress(OSError):
        path.rmdir()


def unlink_quiet(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError, IsADirectoryError, PermissionError):
        path.unlink()


def safe_meta(value: str, *, limit: int = 200) -> str:
    """S3 metadata must be ASCII. Strip rather than reject."""
    return value.encode("ascii", "replace").decode("ascii")[:limit]


def error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None) or {}
    error = response.get("Error") or {}
    return str(error.get("Code", ""))
