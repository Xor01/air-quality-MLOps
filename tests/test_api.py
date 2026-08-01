from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert "model" in body
    assert "threshold" in body
    assert "features" in body
    assert "target" in body

