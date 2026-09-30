import warnings
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.doc_chunk import DocMeta
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChunkingError(Exception):
    """Raised when a chunk exceeds the embedding model limit unexpectedly."""


@dataclass(frozen=True)
class PreparedChunk:
    index: int
    text: str
    token_count: int
    source_text: str
    page_number: int | None


@lru_cache(maxsize=1)
def build_tokenizer() -> BaseTokenizer:
    spec = settings.embedding_spec
    if spec.provider == "openai":
        encoding = tiktoken.get_encoding(spec.tokenizer_id)
        return OpenAITokenizer(tokenizer=encoding, max_tokens=spec.effective_max_tokens)

    tok = HuggingFaceTokenizer.from_pretrained(
        model_name=spec.tokenizer_id,
        max_tokens=spec.effective_max_tokens,
    )
    underlying = tok.tokenizer
    if spec.max_tokens > underlying.model_max_length:
        raise ValueError(f"{spec.name}: spec max_tokens={spec.max_tokens} exceeds tokenizer model_max_length={underlying.model_max_length}")
    return tok


def build_chunker(
    tokenizer: BaseTokenizer | None = None,
) -> HybridChunker:
    return HybridChunker(
        tokenizer=tokenizer or build_tokenizer(),
        merge_peers=settings.chunk_merge_peers,
    )


def first_page_number(meta: DocMeta) -> int | None:
    """Return the smallest page number referenced by this chunk's doc items."""
    pages: set[int] = set()
    for item in meta.doc_items:
        for prov in getattr(item, "prov", []):
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(page_no)
    return min(pages) if pages else None


def _get_captions(meta) -> list[str]:
    """Access meta.captions without triggering Pydantic's deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        caps = getattr(meta, "captions", None)
    return list(caps) if caps else []


def extract_heading_prefix(
    contextual_text: str,
    docling_chunk,
) -> str:
    """
    Extract the prefix that contextualize() prepended to the chunk body.

    Primary path: string suffix match — exact by construction, no
    assumptions about how contextualize() formats headings/captions.

    Fallback: reconstruct from meta.headings / meta.captions in case
    the body was normalized (whitespace trimmed, newlines changed)
    and is no longer a literal suffix of the contextualized text.
    """
    source_text = docling_chunk.text

    if contextual_text.endswith(source_text):
        return contextual_text[: len(contextual_text) - len(source_text)]

    # Fallback — reconstruct from metadata.
    meta = docling_chunk.meta
    parts: list[str] = []

    if meta.headings:
        parts.extend(meta.headings)

    caps = _get_captions(meta)
    if caps:
        parts.extend(caps)

    if parts:
        candidate = "\n".join(parts) + "\n"
        if contextual_text.startswith(candidate):
            return candidate

    # Both paths failed — the body was modified in a way we can't
    # reproduce. Prefix is lost for this chunk. Log so it's visible.
    logger.warning(
        "Could not extract heading prefix; contextualized body is not a "
        "suffix of chunk.text and metadata reconstruction does not match. "
        "Chunk will embed without heading context."
    )
    return ""
