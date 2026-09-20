import pytest_asyncio
from pathlib import Path
from dotenv import load_dotenv

# tests → backend → docmind-ai
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "envs" / "test.env", override=True)

from app.core.config import settings
from app.db import Base, get_db
from app.main import app
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


@pytest_asyncio.fixture(scope="session")
async def engine():
    print("DB URL:", settings.database_url)     # add this line

    eng = create_async_engine(
        settings.database_url,
        poolclass=NullPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await eng.dispose()


@pytest_asyncio.fixture
async def db(engine):
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db):
    async def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as c:
            yield c

    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def authed(client: AsyncClient):
    await client.post(
        url="/api/v1/auth/register",
        json={
            "email": "test@test.com",
            "username": "test",
            "password": "TestPassword_1",
        },
    )
    res = await client.post(
        url="/api/v1/auth/login",
        json={"email": "test@test.com", "password": "TestPassword_1"},
    )
    token = res.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client



