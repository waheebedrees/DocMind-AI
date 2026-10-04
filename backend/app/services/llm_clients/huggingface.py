import asyncio
import threading
from typing import Any

from app.core.exceptions import (
    DocMindError,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from app.core.logging import get_logger
from app.services.llm_clients.model_cache import ModelKey, get_or_load

log = get_logger(__name__)


def _load_embedder(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _get_cached_embedder(model_name: str, device: str) -> tuple[Any, threading.Lock]:
    key = ModelKey(kind="embedder", model_name=model_name, device=device)

    def loader() -> Any:
        try:
            return _load_embedder(model_name, device)
        except (ValueError, KeyError) as exc:
            raise PermanentEmbeddingError(f"cannot load embedder {model_name!r}: {exc}") from exc
        except OSError as exc:
            raise TransientEmbeddingError(f"embedder load I/O failed for {model_name!r}: {exc}") from exc
        except Exception as exc:
            raise TransientEmbeddingError(f"embedder load failed for {model_name!r}: {exc}") from exc

    return get_or_load(key, loader)


class HuggingFaceClient:
    """Stateless wrapper. All state (model, encode lock) lives in the
    module cache, so constructing this is cheap and thread-safe.
    """

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int,
        max_batch_size: int,
        query_prefix: str = "",
        passage_prefix: str = "",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._max_batch_size = max_batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._device = device

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts, self._passage_prefix)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._embed_sync, [text], self._query_prefix)
        return vectors[0]

    def _embed_sync(self, texts: list[str], prefix: str) -> list[list[float]]:
        """Encode a batch of texts with the given prefix.

        Raises PermanentEmbeddingError for input/model-contract problems
        and TransientEmbeddingError for anything a retry might fix.
        """
        if not texts:
            return []

        prepared = [f"{prefix}{t}" if prefix else t for t in texts]
        model, encode_lock = _get_cached_embedder(self._model_name, self._device)

        try:
            with encode_lock:
                vectors = model.encode(
                    prepared,
                    batch_size=min(len(prepared), self._max_batch_size),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except DocMindError:
            # _get_cached_model already classified the load failure.
            raise
        except (TimeoutError, OSError, MemoryError) as exc:
            raise TransientEmbeddingError(f"encode failed (resource): {exc}") from exc
        except RuntimeError as exc:
            raise TransientEmbeddingError(f"encode failed (runtime): {exc}") from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise PermanentEmbeddingError(f"encode failed (input): {exc}") from exc
        except Exception as exc:
            raise TransientEmbeddingError(f"encode failed (unclassified): {exc}") from exc

        out = vectors.tolist()

        if len(out) != len(prepared):
            raise PermanentEmbeddingError(f"model returned {len(out)} vectors for {len(prepared)} inputs")
        for i, v in enumerate(out):
            if len(v) != self._dimension:
                raise PermanentEmbeddingError(f"dimension mismatch on vector {i}: got {len(v)}, expected {self._dimension}")
        return out
