from uuid import UUID


def _key(user_id: UUID, document_id: UUID, name: str) -> str:
    return f"{user_id}/_pipeline/{document_id}/{name}"


def parsed_key(user_id: UUID, document_id: UUID) -> str:
    return _key(user_id, document_id, "parsed.json.gz")


def clean_key(user_id: UUID, document_id: UUID) -> str:
    return _key(user_id, document_id, "clean.json.gz")


def chunks_key(user_id: UUID, document_id: UUID) -> str:
    return _key(user_id, document_id, "chunks.json.gz")


def embedded_chunks_key(user_id: UUID, document_id: UUID) -> str:
    return _key(user_id, document_id, "embedded_chunks.json.gz")


PIPELINE_ARTIFACT_KEYS = (parsed_key, clean_key, chunks_key, embedded_chunks_key)


def all_artifact_keys(user_id: UUID, document_id: UUID) -> list[str]:
    return [fn(user_id, document_id) for fn in PIPELINE_ARTIFACT_KEYS]
