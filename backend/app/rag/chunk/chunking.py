from collections.abc import Iterator
from typing import cast

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.doc_chunk import DocMeta
from docling_core.types.doc import DoclingDocument

from app.core.config import settings

from .docling_chunk import ChunkingError, PreparedChunk, build_chunker, build_tokenizer, extract_heading_prefix, first_page_number
from .splitter import EmbeddingTextSplitter


def chunk_document(
    document: DoclingDocument,
    chunker: HybridChunker | None = None,
) -> Iterator[PreparedChunk]:
    """
    Chunk an already-extracted DoclingDocument.

    Raises:
        ChunkingError: if a produced chunk exceeds the embedding model limit.
    """
    ch = chunker or build_chunker()
    limit = settings.embedding_spec.max_tokens

    for index, docling_chunk in enumerate(ch.chunk(dl_doc=document)):
        text = ch.contextualize(chunk=docling_chunk)
        # we will handel later  page_number
        meta: DocMeta = cast(DocMeta, docling_chunk.meta)
        page_number = first_page_number(meta)
        token_count = ch.tokenizer.count_tokens(text=text)

        if token_count > limit:
            raise ChunkingError(f"Chunk {index} exceeds embedding limit: {token_count} > {limit}. Increase SAFETY_RESERVE.")

        yield PreparedChunk(
            index=index,
            text=text,
            page_number=page_number,
            token_count=token_count,
            source_text=docling_chunk.text,
        )


def chunk_with_splitter(
    document: DoclingDocument,
    chunker: HybridChunker | None = None,
) -> Iterator[PreparedChunk]:
    """
    Chunk an already-extracted DoclingDocument.

    Structural chunks come from HybridChunker; if a chunk (with its
    contextualized heading prefix) exceeds the embedding model's
    content budget, EmbeddingTextSplitter breaks the body into
    token-aligned pieces that still carry the prefix.

    Raises:
        ChunkingError: if a produced chunk exceeds the embedding
        model's effective content budget.
    """
    spec = settings.embedding_spec
    model_tokenizer = build_tokenizer()  # configured with effective_max_tokens

    splitter = EmbeddingTextSplitter(
        model_tokenizer.get_tokenizer(),
        content_budget=spec.content_budget,
        overlap_tokens=spec.overlap_tokens,  # new field on the spec
    )

    chunker = chunker or build_chunker(model_tokenizer)
    output_index = 0

    for docling_chunk in chunker.chunk(dl_doc=document):
        contextual_text = chunker.contextualize(chunk=docling_chunk)
        meta: DocMeta = cast(DocMeta, docling_chunk.meta)
        page_number = first_page_number(meta)
        prefix = extract_heading_prefix(contextual_text, docling_chunk)
        body = contextual_text[len(prefix) :] if prefix else contextual_text

        for embedding_text in splitter.split(body, prefix=prefix):
            model_token_count = splitter.count_model_tokens(embedding_text)

            if model_token_count > spec.effective_max_tokens:
                raise ChunkingError(f"Embedding chunk exceeds model limit: {model_token_count} > {spec.effective_max_tokens}")

            yield PreparedChunk(
                index=output_index,
                text=embedding_text,
                token_count=model_token_count,
                page_number=page_number,
                source_text=docling_chunk.text,
            )
            output_index += 1
