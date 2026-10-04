from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str  # canonical id; also the dict key
    provider: Literal["huggingface", "openai"]
    dimension: int
    tokenizer_id: str  # HF repo id or tiktoken encoding
    max_tokens: int  # model's hard input limit
    max_batch_size: int  # max inputs per request
    max_batch_tokens: int | None = None  # max tokens per request (OpenAI: 300_000)
    reserve: int = 64  # special tokens + BPE boundary margin
    overlap_tokens: int = 32  # adjacent chunks share this many tokens
    # ← new: [CLS], [SEP] for BERT-family; 0 for OpenAI
    special_tokens: int = 2

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    query_prefix: str = ""
    passage_prefix: str = ""
    embed_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"{self.name}: max_tokens must be positive")
        if self.dimension <= 0:
            raise ValueError(f"{self.name}: dimension must be positive")
        if self.max_batch_size <= 0:
            raise ValueError(f"{self.name}: max_batch_size must be positive")
        if not 0 <= self.reserve < self.max_tokens:
            raise ValueError(f"{self.name}: reserve ({self.reserve}) must be in [0, max_tokens={self.max_tokens})")

    @property
    def effective_max_tokens(self) -> int:
        """Content budget after reserving safety margin."""
        return self.max_tokens - self.reserve

    @property
    def content_budget(self) -> int:
        """Content budget after reserving safety margin."""
        return self.effective_max_tokens - self.special_tokens


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    spec.name: spec
    for spec in [
        EmbeddingModelSpec(
            name="BAAI/bge-small-en-v1.5",
            provider="huggingface",
            dimension=384,
            tokenizer_id="BAAI/bge-small-en-v1.5",
            max_tokens=512,
            max_batch_size=64,
            reserve=64,
            query_prefix="Represent this sentence for searching relevant passages: ",
            passage_prefix="",
            embed_timeout_s=50,
        ),
        EmbeddingModelSpec(
            name="sentence-transformers/all-MiniLM-L6-v2",
            provider="huggingface",
            dimension=384,
            tokenizer_id="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=512,
            max_batch_size=64,
            reserve=64,
            query_prefix="Represent this sentence for searching relevant passages: ",
            passage_prefix="",
            embed_timeout_s=50,
        ),
        EmbeddingModelSpec(
            name="BAAI/bge-base-en-v1.5",
            provider="huggingface",
            dimension=768,
            tokenizer_id="BAAI/bge-base-en-v1.5",
            max_tokens=512,
            max_batch_size=64,
            reserve=64,
            query_prefix="Represent this sentence for searching relevant passages: ",
            passage_prefix="",
            embed_timeout_s=50,
        ),
        EmbeddingModelSpec(
            name="text-embedding-3-small",
            provider="openai",
            dimension=1536,
            tokenizer_id="cl100k_base",
            max_tokens=8191,
            special_tokens=0,
            max_batch_size=2048,
            max_batch_tokens=300_000,
            reserve=16,
            embed_timeout_s=10,
        ),
        EmbeddingModelSpec(
            name="text-embedding-3-large",
            provider="openai",
            special_tokens=0,
            dimension=3072,
            tokenizer_id="cl100k_base",
            max_tokens=8191,
            max_batch_size=2048,
            max_batch_tokens=300_000,
            reserve=16,
            embed_timeout_s=10,
        ),
    ]
}


def get_spec(model_name: str) -> EmbeddingModelSpec:
    try:
        return EMBEDDING_MODELS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown embedding model '{model_name}'. Supported: {sorted(EMBEDDING_MODELS)}") from exc
