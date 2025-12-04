from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

CODE_SUCCESS = 200


def test_predict_empty():
    response = client.post("/predict", json={"text": "Hello world"})
    assert response.status_code == CODE_SUCCESS
