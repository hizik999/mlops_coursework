from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

CODE_SUCCESS = 200


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == CODE_SUCCESS
    data = response.json()
    assert "status" in data
