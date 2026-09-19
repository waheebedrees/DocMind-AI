from app.db.base import Base
import app.models  # noqa: F401


EXPECTED_TABLES = {
    "users",
    "documents",
    "document_chunks",
    "conversations",
    "messages",
    "citations",
    "processing_jobs",
    "extractions",
    "evaluation_runs",
    "evaluation_results",
}


def test_all_models_registered() -> None:
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES
