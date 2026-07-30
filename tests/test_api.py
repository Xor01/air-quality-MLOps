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


def test_predict():
    payload = {
        "pm2_5": 25.0,
        "pm10": 40.0,
        "temperature_2m": 35.0,
        "relative_humidity_2m": 30.0,
        "wind_speed_10m": 10.0,
        "hour": 14,
        "day_of_week": 2,
        "pm2_5_lag_1": 24.0,
        "pm2_5_lag_3": 23.0,
        "pm2_5_rolling_mean_6": 24.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "prediction_probability" in body
    assert "model" in body
    assert "threshold" in body