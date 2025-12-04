from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    # статус может быть "ok" (локально с моделью) или "model_not_loaded" (в CI)
    assert "status" in data
    assert data["status"] in ("ok", "model_not_loaded")
