from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    provider: Literal["local", "openai"]
    dimension: int
    tokenizer: str  # HF repo id or tiktoken encoding
    max_tokens: int  # model's max input length
    max_batch_size: int  # provider's per-request limit


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    # Local, free, offline
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-small-en-v1.5",
        provider="local",
        dimension=384,
        tokenizer="BAAI/bge-small-en-v1.5",
        max_tokens=512,
        max_batch_size=64,
    ),
    "all-MiniLM-L6-v2": EmbeddingModelSpec(
        name="all-MiniLM-L6-v2",
        provider="local",
        dimension=384,
        tokenizer="all-MiniLM-L6-v2",
        max_tokens=512,
        max_batch_size=64,
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-base-en-v1.5",
        provider="local",
        dimension=768,
        tokenizer="BAAI/bge-base-en-v1.5",
        max_tokens=512,
        max_batch_size=64,
    ),
    # OpenAI, Phase 2 -
    "text-embedding-3-small": EmbeddingModelSpec(
        name="text-embedding-3-small",
        provider="openai",
        dimension=1536,
        tokenizer="cl100k_base",
        max_tokens=8191,
        max_batch_size=2048,
    ),
    "text-embedding-3-large": EmbeddingModelSpec(
        name="text-embedding-3-large",
        provider="openai",
        dimension=3072,
        tokenizer="cl100k_base",
        max_tokens=8191,
        max_batch_size=2048,
    ),
}


def get_spec(model_name: str) -> EmbeddingModelSpec:
    try:
        return EMBEDDING_MODELS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown embedding model '{model_name}'. Supported: {sorted(EMBEDDING_MODELS)}") from exc
