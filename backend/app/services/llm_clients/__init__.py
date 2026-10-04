from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.services.llm_clients.base import EmbeddingClient
from app.services.llm_clients.huggingface import HuggingFaceClient


@lru_cache(maxsize=2)
def get_hf_embed() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_spec.name)


@lru_cache(maxsize=2)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.embedding_spec.cross_encoder_model)


@lru_cache(maxsize=1)
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
