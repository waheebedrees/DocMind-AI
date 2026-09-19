# DocMind AI

Production-style RAG knowledge assistant — ingestion, retrieval, citations,
conversations, evaluation, and observability.

## Stack
- **Backend:** FastAPI + Python 3.12
- **Database:** PostgreSQL 16 + pgvector
- **Cache/Queue:** Redis 7
- **Container:** Docker Compose

```bash
cp .env.example .env
make up
curl http://localhost:8000/api/v1/health
```

## Dev workflow

```bash
make logs          # tail backend
make shell         # exec into backend container
make test          # run pytest
make lint          # ruff + mypy
make lint-check    # CI without fix
make fmt           # auto-format
```

## Roadmap

- [x] Phase 1 · Step 1 — Scaffold, Docker, health
- [x] Phase 1 · Step 2 — Models + Alembic migrations
