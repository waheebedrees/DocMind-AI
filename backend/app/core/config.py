from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    debug: bool = False
    version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    app_name: str = "DocMind AI"
    api_v1_prefix: str = "/api/v1"
    # CORS
    allow_credentials: bool = True
    api_key_headers: str = "X-API-KEY"
    allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE"]

    # Database
    POSTGRES_USER: str = "docmind"
    POSTGRES_PASSWORD: str = "docmind"
    POSTGRES_PORT: int = 5432
    # ── Resolved at access time (see properties below) ────────────
    POSTGRES_HOST: str | None = None  # None → auto-detect
    POSTGRES_DB: str | None = None  # None → default per env

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # Redis
    redis_url: str = "redis://redis:6379/0"
    redis_password: str = "REDIS_PASSWORD"

    # Security (used from step 3 onward, kept here so /health can report shape)
    jwt_secret: str = "secret" * 6
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # LLM (used from step 5 onward)
    llm_provider: Literal["openai", "anthropic"] = "openai"
    embedding_dim: int = 1536

    @property
    def allow_origins(self):
        return ["*"] if self.environment == "development" else []

    @property
    def docs_url(self):
        return "/docs" if settings.environment != "production" else None

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
