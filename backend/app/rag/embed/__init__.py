from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.rag.embed.embed import embed_in_batches


@lru_cache(maxsize=2)
def get_hf_embed() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_spec.name)


@lru_cache(maxsize=2)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.embedding_spec.cross_encoder_model)


__all__ = ["embed_in_batches"]
