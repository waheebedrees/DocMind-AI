import asyncio
import threading
from functools import lru_cache
from typing import Protocol, runtime_checkable

from app.core.config import settings
from app.core.decorator import with_async_backoff
from app.core.logging import get_logger

log = get_logger(__name__)


class PipelineError(Exception):
    """ "Base class for Rag pipeline"""


class EmbeddingError(PipelineError):
    """Base class for embedding failures."""


class TransientEmbeddingError(EmbeddingError):
    """Rate limit, network, timeout — retry."""


class PermanentEmbeddingError(EmbeddingError):
    """Invalid input, wrong dimension, model not found — do not retry."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Provider-agnostic embedding interface.

    Implementations must:
      * Return one vector per input text, in order.
      * Use the correct instruction prefix for queries vs passages.
      * Normalize vectors for cosine similarity.
      * Raise TransientEmbeddingError on retryable failures.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def max_batch_size(self) -> int: ...

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks. No instruction prefix."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query. Uses the query prefix."""
        ...


async def embed_in_batches(
    client: EmbeddingClient,
    texts: list[str],
    *,
    concurrency: int = 4,
) -> list[list[float]]:
    """Embed texts in batches with bounded concurrency and retry.

    Preserves input order in the output.
    """
    if not texts:
        return []

    batch_size = client.max_batch_size
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(concurrency)

    async def one_batch(batch: list[str], idx: int) -> list[list[float]]:
        async with semaphore:
            return await _embed_with_retry(client, batch, batch_index=idx)

    log.info(
        "embedding_started",
        total_texts=len(texts),
        batches=len(batches),
        batch_size=batch_size,
        concurrency=concurrency,
    )

    results = await asyncio.gather(*[one_batch(b, i) for i, b in enumerate(batches)])

    flat = [vec for batch in results for vec in batch]
    if len(flat) != len(texts):
        raise RuntimeError(f"embedding count mismatch: {len(flat)} vectors for {len(texts)} texts")
    return flat


@with_async_backoff(max_retries=4, initial_delay=1.0, backoff_factor=2.0, max_delay=30.0)
async def _embed_with_retry(
    client: EmbeddingClient,
    batch: list[str],
    *,
    batch_index: int,
) -> list[list[float]]:
    try:
        return await client.embed_passages(batch)
    except Exception as e:
        log.warning("embedding_retry", batch_index=batch_index, e=str(e))
        raise TransientEmbeddingError("embedding failed") from e


class HuggingFaceClient:
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

        self._model = None  # lazy
        self._lock = threading.Lock()

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

    def _load(self):
        if self._model is None:
            log.info("loading_embedding_model", model=self._model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            log.info("embedding_model_loaded", model=self._model_name)
        return self._model

    def _embed_sync(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []

        prepared = [f"{prefix}{t}" if prefix else t for t in texts]

        try:
            with self._lock:
                model = self._load()
                vectors = model.encode(
                    prepared,
                    batch_size=min(len(prepared), self._max_batch_size),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except RuntimeError as exc:
            raise TransientEmbeddingError(f"local embed failed: {exc}") from exc
        except Exception as exc:
            raise PermanentEmbeddingError(f"local embed error: {exc}") from exc

        out = vectors.tolist()
        for v in out:
            if len(v) != self._dimension:
                raise PermanentEmbeddingError(f"dimension mismatch: model returned {len(v)}, expected {self._dimension}")
        return out


@lru_cache
def get_embedder() -> EmbeddingClient:
    spec = settings.embedding_spec

    if spec.provider == "huggingface":
        return HuggingFaceClient(
            spec.name,
            dimension=spec.dimension,
            max_batch_size=spec.max_batch_size,
            query_prefix=spec.query_prefix,
            passage_prefix=spec.passage_prefix,
        )

    if spec.provider == "openai":
        raise NotImplementedError("OpenAI embeddings land in Phase 2. Set EMBEDDING_MODEL to a local model.")

    raise ValueError(f"Unknown embedding provider: {spec.provider}")
