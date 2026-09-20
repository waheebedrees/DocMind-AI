from app.api.v1 import health as health_module
from app.main import app
from httpx import ASGITransport, AsyncClient


async def __db_ok(db) -> health_module.ComponentHealth:
    return health_module.ComponentHealth(status="ok")


async def __db_fail(db) -> health_module.ComponentHealth:
    return health_module.ComponentHealth(status="error", detail="boom")


async def _ok() -> health_module.ComponentHealth:
    return health_module.ComponentHealth(status="ok")


async def test_health_ok_when_all_dependencies_up(monkeypatch):
    monkeypatch.setattr(health_module, "_check_postgres", __db_ok)
    monkeypatch.setattr(health_module, "_check_redis", _ok)

    # print(f"DEBUG: {sys.path}")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["postgres"]["status"] == "ok"
    assert body["components"]["redis"]["status"] == "ok"


async def test_health_degraded_when_dependency_down(monkeypatch):
    monkeypatch.setattr(health_module, "_check_postgres", __db_fail)
    monkeypatch.setattr(health_module, "_check_redis", _ok)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "error"
    assert body["components"]["postgres"]["detail"] == "boom"
