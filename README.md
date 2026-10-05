# DocMind AI

Production-grade RAG knowledge assistant — ingestion, retrieval,
citations, conversations, evaluation, and observability.

## Stack

- **Backend:** FastAPI + Python 3.12
- **Database:** PostgreSQL 16 + pgvector
- **Cache/Queue:** Redis 7 + arq
- **Extraction:** Docling (PDF, DOCX, PPTX, XLSX, HTML, MD, images)
- **Embeddings:** sentence-transformers (`BAAI/bge-small-en-v1.5`, 384-dim)
- **Storage:** local filesystem or S3-compatible (MinIO, R2, GCS-S3)
- **Container:** Docker Compose

## Quick start

```bash
cp envs/dev.env.example envs/dev.env   # fill in secrets
make up
make upgrade                           # apply migrations
curl http://localhost:8000/api/v1/health
```

## Dev workflow

```bash
make logs          # tail backend
make shell         # exec into backend container
make db-shell      # psql into postgres
make test          # run pytest
make lint          # ruff + mypy
make lint-check    # CI without fix
make fmt           # auto-format
make history       # alembic migration history
make current       # alembic current revision
```

## Project layout

```
backend/
  app/
    api/
      middleware.py
      v1/
        auth.py
        documents.py
        health.py
        retrieval.py
        router.py
    core/
      arq.py
      auth.py
      config.py
      decorator.py
      deps.py
      embedding_models.py
      exceptions.py
      logging.py
      retrieval_settings.py
    db/
      base.py
      redis.py
      session.py
      repositories/
        base.py
        chunks.py
        citation_repo.py
        conversation_repo.py
        documents.py
        evaluation_repo.py
        extraction_repo.py
        jobs.py
        message_repo.py
        user_repo.py
    models/
      chunk.py
      citation.py
      conversation.py
      document.py
      enums.py
      evaluation.py
      extraction.py
      message.py
      mixins.py
      processing_job.py
      user.py
    rag/
      cleaning.py
      extraction.py
      ingestion.py
      chunk/
        chonkie_chunk.py
        chunking.py
        docling_chunk.py
        splitter.py
        types.py
      embed/
        embed.py
      retrieval/
        context.py
        ranking.py
        rerank.py
        retrieval.py
        types.py
    schemas/
      auth.py
      document.py
      user.py
    services/
      document_service.py
      user_service.py
      llm_clients/
        base.py
        huggingface.py
        model_cache.py
        rerank/
          factory.py
          huggingface_rerank.py
      storage/
        _fn.py
        base.py
        keys.py
        local.py
        mime.py
        s3.py
        uploads.py
    workers/
      settings.py
      sweeper.py
      tasks.py
  alembic/
    env.py
    script.py.mako
    versions/
      965b0a74c3ea_initial_schema.py
      bdbcbbe5489d_fixing_citatoin_table.py
      459af247767d_add_unique_constraint_to_citations.py
  tests/
    conftest.py
    test_health.py
    test_models_importable.py
    integration/
    units/
      rag/
      repos/
      storage/
      workers/
```

## Testing

```bash
make test
```

Tests live under `backend/tests/units/`, grouped by domain:

- `units/rag/` — retrieval (RRF, MMR), context assembly
- `units/repos/` — repositories
- `units/storage/` — local and S3 backends, MIME detection, uploads
- `units/workers/` — ingestion pipeline stages and the maintenance sweeper

`backend/tests/integration/` holds opt-in tests that need a real
database or S3-compatible endpoint.

## Roadmap

- [x] Phase 1 · Step 1 — Scaffold, Docker, health
- [x] Phase 1 · Step 2 — Models + Alembic migrations
- [x] Phase 2 · Step 1 — Auth (register, login, JWT, protected routes)
- [x] Phase 2 · Step 2 — Storage layer (local + S3, content-addressed dedup)
- [x] Phase 2 · Step 3 — Document extraction (Docling) and chunking
- [x] Phase 2 · Step 4 — Arq worker scaffolding
- [x] Phase 2 · Step 5 — Upload endpoint + ingest worker
- [x] Phase 2 · Step 6 — Embedding pipeline + pgvector persistence
- [x] Phase 3 · Step 1 — Retrieval (hybrid search)
- [ ] Phase 3 · Step 2 — RAG answer with citations