from chonkie import TokenChunker
from chonkie.types import Chunk

from app.core.config import settings


def get_chonkie_chunks(markdown_text: str) -> list[Chunk]:
    """
    Converts a document with Docling and chunks it with Chonkie
    using a tokenizer that matches the embedding model.
    """
    # 1. Get the embedding model spec to know the tokenizer and limit
    spec = settings.embedding_spec

    # 3. Initialize Chonkie's TokenChunker
    #    - tokenizer: Pass the HF tokenizer instance directly
    #    - chunk_size: Set to the model's max_tokens (e.g., 512)
    #    - chunk_overlap: A sensible default, e.g., 10-20% of chunk_size
    chunker = TokenChunker(
        tokenizer=spec.tokenizer_id,
        chunk_size=spec.max_tokens,
        chunk_overlap=50,  # Adjust as needed
    )

    # 5. Chunk the Markdown text
    chunks = chunker.chunk(markdown_text)

    return chunks
