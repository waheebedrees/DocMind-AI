from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

load_dotenv(str(PROJECT_ROOT / "envs" / "dev.env"))
from backend.app.core.logging import get_logger
from backend.app.core.config import settings


logger = get_logger(__name__)

model = SentenceTransformer(settings.embedding_spec.name)


def check_chunks(chunks, spec, *, use_model_tokenizer: bool) -> None:
    """Three-level verification of a chunk list against the embedding spec."""
    assert chunks, "no chunks produced"

    # Level 1 — pipeline's own accounting
    worst = max(c.token_count for c in chunks)
    assert worst <= spec.effective_max_tokens, (
        f"level 1 failed: {worst} > {spec.effective_max_tokens}"
    )
    print(f"[1] pipeline self-check : {worst}/{spec.effective_max_tokens}")

    # Level 2 — independent HF tokenizer
    from transformers import AutoTokenizer
    checker = AutoTokenizer.from_pretrained(spec.tokenizer_id)

    worst2 = 0
    for c in chunks:
        n = len(checker.encode(c.text, add_special_tokens=True, truncation=False))
        worst2 = max(worst2, n)
        assert n <= spec.effective_max_tokens, (
            f"level 2 failed: chunk {c.index} = {n} tokens (> {spec.effective_max_tokens})"
        )
    print(f"[2] independent tokenizer: {worst2}/{spec.effective_max_tokens}")

    # Level 3 — actual embedder accepts everything
    texts = [c.text for c in chunks]
    print(f"[3] calling model.encode on {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=spec.max_batch_size,
        show_progress_bar=True,
    )
    print("[3] model.encode returned")

    assert embeddings.shape == (len(texts), spec.dimension), (
        f"unexpected embedding shape: {embeddings.shape}"
    )

    model_tok = model.tokenizer
    for i, text in enumerate(texts):
        n = len(model_tok.encode(text, add_special_tokens=True))
        assert n <= spec.effective_max_tokens, (
            f"level 3 failed: chunk {i} = {n} tokens"
        )

    print(f"[3] model accepted all {len(texts)} chunks → {embeddings.shape}")
    print(f"\n✅ {len(chunks)} chunks, max {worst2}/{spec.effective_max_tokens}")


def main() -> None:
    source = "./docs/ZeroStrike3.docx"
    from backend.app.rag.chunk.chunking import chunk_document,  chunk_with_splitter
    from backend.app.rag.extraction import extract_document

    document = extract_document(source)
    chunks = list(chunk_document(document=document))
    check_chunks(chunks, settings.embedding_spec, use_model_tokenizer=True)

    chunks = list(chunk_with_splitter(document=document))
    check_chunks(chunks, settings.embedding_spec, use_model_tokenizer=True)


if __name__ == "__main__":
    main()
