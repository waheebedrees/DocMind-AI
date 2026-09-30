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

## Roadmap

- [x] Phase 1 · Step 1 — Scaffold, Docker, health
- [x] Phase 1 · Step 2 — Models + Alembic migrations
- [x] Phase 2 · Step 1 — Auth (register, login, JWT, protected routes)
- [x] Phase 2 · Step 2 — Storage layer (local + S3, content-addressed dedup)
- [x] Phase 2 · Step 3 — Document extraction (Docling) and chunking
- [x] Phase 2 · Step 4 — Arq worker scaffolding
- [ ] Phase 2 · Step 5 — Upload endpoint + ingest worker
- [ ] Phase 2 · Step 6 — Embedding pipeline + pgvector persistence
- [ ] Phase 3 · Step 1 — Retrieval (hybrid search)
- [ ] Phase 3 · Step 2 — RAG answer with citations