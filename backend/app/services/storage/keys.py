import hashlib
from functools import lru_cache
from importlib.metadata import version

from app.core.config import settings

_STAGE_ORDER = ("extract", "clean", "chunk", "embed", "index")


def _stage_config(stage: str) -> list[str]:
    """Everything that can change this stage's output. Bump the version
    field when behavior changes in a way config can't capture (a code fix)."""
    spec = settings.embedding_spec
    pipe = settings.pipeline

    match stage:
        case "extract":
            return [
                f"version={pipe.extract_version}",
                f"docling={version('docling')}",
            ]
        case "clean":
            return [f"version={pipe.clean_version}"]
        case "chunk":
            return [
                f"version={pipe.chunk_version}",
                f"max_tokens={spec.max_tokens}",
                f"overlap={spec.overlap_tokens}",
                f"content_budget={spec.content_budget}",
            ]
        case "embed":
            return [
                f"version={pipe.embed_version}",
                f"model={spec.name}",
                f"dim={spec.dimension}",
                f"sentence_transformers={version('sentence-transformers')}",
            ]
        case "index":
            return [f"version={pipe.index_version}"]
    raise ValueError(f"unknown stage {stage!r}")


@lru_cache(maxsize=1)
def _fingerprints() -> dict[str, str]:
    """One hash per stage. Each stage's fingerprint includes the previous
    stage's fingerprint, so bumping an upstream version changes every
    downstream key and leaves upstream caches intact.

    Example: bump clean_version -> extract cache hit, clean/chunk/embed/
    index all recompute on the next pipeline run.
    """
    out: dict[str, str] = {}
    chain = ""
    for stage in _STAGE_ORDER:
        material = f"{chain}|{'|'.join(_stage_config(stage))}"
        chain = hashlib.sha256(material.encode()).hexdigest()[:16]
        out[stage] = chain
    return out


def _cache_key(content_hash: str, stage: str) -> str:
    fp = _fingerprints()[stage]
    return f"_pipeline/cache/{fp}/{content_hash}/{stage}.json.gz"


def parsed_key(content_hash: str) -> str:
    return _cache_key(content_hash, "extract")


def clean_key(content_hash: str) -> str:
    return _cache_key(content_hash, "clean")


def chunks_key(content_hash: str) -> str:
    return _cache_key(content_hash, "chunk")


def embedded_chunks_key(content_hash: str) -> str:
    return _cache_key(content_hash, "embed")


def artifact_keys_from(content_hash: str, from_stage: str) -> list[str]:
    """Cache keys for from_stage and every stage downstream of it.

    Used by reprocess(): deleting these forces recomputation from
    from_stage onward while keeping upstream artifacts cached.
    """
    idx = _STAGE_ORDER.index(from_stage)
    return [_cache_key(content_hash, s) for s in _STAGE_ORDER[idx:]]


def all_artifact_keys(content_hash: str) -> list[str]:
    return [_cache_key(content_hash, s) for s in _STAGE_ORDER]
