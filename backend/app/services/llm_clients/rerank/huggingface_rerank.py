"""Cross-encoder reranker backed by sentence-transformers.

Runs on CPU by default. The model is loaded once per process and
serialized with a per-model lock, because CrossEncoder.predict is
synchronous, CPU-bound, and not documented as thread-safe.
"""

import asyncio
import threading
from typing import Any

import numpy as np
from app.core.exceptions import (
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from app.core.logging import get_logger
from app.services.llm_clients.model_cache import ModelKey, get_or_load

log = get_logger(__name__)


def _load_reranker(model_name: str, device: str, max_length: int) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device, max_length=max_length)


def _get_cached_reranker(model_name: str, device: str, max_length: int) -> tuple[Any, threading.Lock]:
    key = ModelKey(
        kind="reranker",
        model_name=model_name,
        device=device,
        max_length=max_length,
    )

    def loader() -> Any:
        try:
            return _load_reranker(model_name, device, max_length)
        except (ValueError, KeyError) as exc:
            raise PermanentEmbeddingError(f"cannot load reranker {model_name!r}: {exc}") from exc
        except OSError as exc:
            raise TransientEmbeddingError(f"reranker load I/O failed for {model_name!r}: {exc}") from exc
        except Exception as exc:
            raise TransientEmbeddingError(f"reranker load failed for {model_name!r}: {exc}") from exc

    return get_or_load(key, loader)


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder on CPU.

    ~200 MB loaded for MiniLM-L-6-v2. predict() is synchronous and
    CPU-bound, so it runs in a thread and is serialized per model.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._device = device

    async def score(self, query: str, texts: list[str]) -> list[float]:
        return await asyncio.to_thread(self._score_sync, query, texts)

    def _score_sync(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []

        pairs = [(query, t) for t in texts]
        model, encode_lock = _get_cached_reranker(self._model_name, self._device, self._max_length)

        try:
            with encode_lock:
                raw = model.predict(
                    pairs,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        except (TimeoutError, OSError) as exc:
            raise TransientEmbeddingError(f"reranker encode failed (resource): {exc}") from exc
        except MemoryError as exc:
            # Same input, same process, same batch size on retry —
            # this is not going to succeed by trying again. Callers
            # that want to retry with a smaller batch must do so
            # explicitly (shrink, then re-invoke).
            raise PermanentEmbeddingError(f"reranker encode OOM at batch_size=16: {exc}") from exc
        except RuntimeError as exc:
            # Torch runtime errors are usually recoverable (device
            # busy, transient allocator state). Retry.
            raise TransientEmbeddingError(f"reranker encode failed (runtime): {exc}") from exc
        except (ValueError, TypeError, KeyError) as exc:
            # Malformed input — retrying with the same input fails
            # the same way.
            raise PermanentEmbeddingError(f"reranker encode failed (input): {exc}") from exc
        except Exception as exc:
            raise TransientEmbeddingError(f"reranker encode failed (unclassified): {exc}") from exc

        arr = np.asarray(raw)
        if arr.ndim != 1:
            # Single-output cross-encoders return shape (N,). Anything
            # else means the model's head does not match this wrapper's
            # assumptions — fail loudly rather than misalign scores.
            raise PermanentEmbeddingError(f"reranker returned shape {arr.shape}, expected 1D")

        return [float(s) for s in arr]
