from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=2)
def get_hf_embed() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_spec.name)


@lru_cache(maxsize=2)
def get_cores_encoder() -> CrossEncoder:
    return CrossEncoder(settings.embedding_spec.cross_encoder_model)
