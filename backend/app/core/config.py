from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.embedding_models import EmbeddingModelSpec, get_spec


class Environment(StrEnum):
    "Application Environment"

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"


class StorageBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


def get_storage_backend() -> StorageBackend:
    import os

    env = os.environ.get("STORAGE_BACKEND", "local")
    if env == "s3":
        return StorageBackend.S3
    return StorageBackend.LOCAL


def get_environment() -> Environment:
    """
    get current app environment
    """
    import os

    env = os.environ.get("APP_ENV", "DEV")
    match env:
        case "prod" | "production":
            return Environment.PRODUCTION
        case "stage" | "staging":
            return Environment.STAGING
        case "test" | "testing":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


class PipelineSettings(BaseModel):
    """Per-stage version pins. Bump one to invalidate that stage and
    everything downstream of it. Extract is the only stage that requires
    re-parsing the source PDF; the rest reuse the prior artifact."""

    extract_version: str = "1"  # docling config, parser swaps
    clean_version: str = "1"  # ToC pruning, normalization rules
    chunk_version: str = "1"  # splitter, contextualization
    embed_version: str = "1"  # batching, prefix handling
    index_version: str = "1"  # chunk row shape, label encoding


class JobSettings(BaseModel):
    max_jobs: int = 4
    job_timeout: int = 600  # 10 min hard ceiling per stage
    max_tries: int = 3
    retry_delay: int = 10  # seconds; ARQ applies exponential backoff
    keep_result: int = 3600  # 1 hour — Redis is transport, not truth
    retry_jobs: bool = True
    health_check_interval: int = 30


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"envs/{get_environment().value}.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    debug: bool = False
    version: str = "0.1.0"
    environment: Environment = Field(default_factory=get_environment)
    app_name: str = "DocMind AI"
    api_v1_prefix: str = "/api/v1"

    # CORS
    api_key_headers: str = "X-H"
    allow_credentials: bool = True
    allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE"]

    # Database
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_PORT: int = 5432
    POSTGRES_HOST: str | None = None
    POSTGRES_DB: str | None = None
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # Redis
    redis_host: str | None = None
    redis_port: int = 6379
    redis_password: str | None = None

    # Security (used from step 3 onward, kept here so /health can report shape)
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"  # has a sane default

    access_token_expire_minutes: int = 60

    refresh_token_expire_days: int = 7

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_max_tokens: int = 512
    chunk_merge_peers: bool = True

    # Storage
    storage_backend: StorageBackend = Field(default_factory=get_storage_backend)
    storage_root: Path = Path("/var/lib/docmind/storage")
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    upload_chunk_bytes: int = 64 * 1024  # 64 KB
    mime_sniff_bytes: int = 8 * 1024  # 8 KB head buffer

    # Quotas (from the gap analysis)
    max_documents_per_user: int = 100
    max_total_bytes_per_user: int = 500 * 1024 * 1024  # 500 MB

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    job_settings: JobSettings = Field(default_factory=JobSettings)

    @property
    def embedding_spec(self) -> EmbeddingModelSpec:
        return get_spec(self.embedding_model)

    @property
    def embedding_dim(self) -> int:
        return self.embedding_spec.dimension

    @property
    def embedding_tokenizer(self) -> str:
        return self.embedding_spec.tokenizer_id

    @property
    def embedding_provider(self) -> str:
        return self.embedding_spec.provider

    @property
    def allow_origins(self):
        return ["*"] if self.environment == "development" else []

    @property
    def docs_url(self):
        return "/docs" if settings.environment != "production" else None

    @model_validator(mode="after")
    def _validate_chunking_fits_model(self) -> "Settings":
        spec = self.embedding_spec
        if self.chunk_max_tokens > spec.max_tokens:
            raise ValueError(
                f"chunk_max_tokens={self.chunk_max_tokens} exceeds {spec.name} max_tokens={spec.max_tokens}. Chunks would be truncated at embed time."
            )
        return self

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        if self.environment == "production" and self.debug:
            raise ValueError("DEBUG must be false in production")
        return self

    @model_validator(mode="after")
    def _validate_storage(self) -> "Settings":
        if self.mime_sniff_bytes < 1024:
            raise ValueError("mime_sniff_bytes must be >= 1024")
        if self.upload_chunk_bytes < 4096:
            raise ValueError("upload_chunk_bytes must be >= 4096")

        if self.storage_backend == "local" and not self.storage_root.is_absolute():
            raise ValueError("storage_root must be an absolute path")
        return self

    @field_validator("storage_root", mode="after")
    @classmethod
    def _ensure_absolute(cls, v: Path) -> Path:
        if v.is_absolute():
            return v

        # backend/app/core -> repo root
        _REPO_ROOT = Path(__file__).resolve().parents[3]
        # On Windows, /var/lib/... is not absolute — anchor it to the repo.
        return _REPO_ROOT / v.relative_to("/") if v.is_absolute() is False and str(v).startswith("/") else _REPO_ROOT / v

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
