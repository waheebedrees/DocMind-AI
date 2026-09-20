from enum import StrEnum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.embedding_models import EmbeddingModelSpec, get_spec


class Environment(StrEnum):
    "Application Environment"

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"


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

    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_max_tokens: int = 512
    chunk_merge_peers: bool = True

    # @computed_field
    @property
    def embedding_spec(self) -> EmbeddingModelSpec:
        return get_spec(self.embedding_model)

    @property
    def embedding_dim(self) -> int:
        return self.embedding_spec.dimension

    @property
    def embedding_tokenizer(self) -> str:
        return self.embedding_spec.tokenizer

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
        if self.environment == "production":
            if self.jwt_secret in {"change-me", "change-me-in-real-env", ""}:
                raise ValueError("JWT_SECRET must be set in production")
            if self.debug:
                raise ValueError("DEBUG must be false in production")
        return self

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
