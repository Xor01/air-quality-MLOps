from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "models/model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["pipeline"]
threshold = float(artifact["threshold"])
model_name = artifact["model_name"]

app = FastAPI(
    title="Riyadh Air Quality API",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    pm2_5: float = Field(..., ge=0)
    pm10: float = Field(..., ge=0)
    temperature_2m: float
    relative_humidity_2m: float = Field(..., ge=0, le=100)
    wind_speed_10m: float = Field(..., ge=0)

    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)

    pm2_5_lag_1: float = Field(..., ge=0)
    pm2_5_lag_3: float = Field(..., ge=0)
    pm2_5_rolling_mean_6: float = Field(..., ge=0)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }

@app.get("/model-info")
def model_info():
    return {
        "model": model_name,
        "threshold": threshold,
        "features": artifact["features"],
        "target": artifact["target"],
        "version": "1.0.0",
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    data = pd.DataFrame([request.model_dump()])

    probability = float(model.predict_proba(data)[0][1])

    prediction = int(probability >= threshold)

    log_entry = {
        "request_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **request.model_dump(),
        "prediction_probability": probability,
        "prediction": prediction,
    }

    log_dir = Path("data/monitoring")
    log_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([log_entry]).to_csv(
        log_dir / "predictions.csv",
        mode="a",
        header=not (log_dir / "predictions.csv").exists(),
        index=False,
    )

    return {
        "prediction": prediction,
        "prediction_probability": round(probability, 4),
        "threshold": threshold,
        "model": model_name,
    }