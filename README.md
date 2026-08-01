# Riyadh Air Quality MLOps

An end-to-end machine learning project that predicts whether PM2.5 pollution in Riyadh will exceed **35 µg/m³ in the next hour**.

The project includes data collection, feature engineering, time-aware model training, MLflow experiment tracking, a FastAPI prediction service, a Streamlit interface, tests, and Docker deployment.

## Workflow

```text
Open-Meteo APIs
      ↓
Raw JSON data
      ↓
Feature engineering
      ↓
Time-series training and evaluation
      ↓
Saved model + MLflow artifacts
      ↓
FastAPI service ← Streamlit interface
```

## Tech stack

- Python 3.10+
- pandas, PyArrow, scikit-learn
- MLflow
- FastAPI and Pydantic
- Streamlit
- pytest and Ruff
- Docker Compose
- uv

## Project structure

```text
air-quality-mlops/
├── app/
│   ├── api/main.py             # FastAPI service
│   └── frontend/app.py         # Streamlit interface
├── src/air_quality/
│   ├── collect.py              # Collects Open-Meteo data
│   ├── features.py             # Builds features and target
│   └── train.py                # Trains and evaluates models
├── data/
│   ├── raw/                    # Raw API data
│   ├── processed/              # Prepared model table
│   ├── monitoring/             # Reference dataset
│   └── predictions/            # API prediction log
├── models/model.joblib         # Trained model bundle
├── reports/evaluation/         # Evaluation reports
├── tests/                      # API tests
├── docker-compose.yml
└── pyproject.toml
```

## Model inputs

The model uses ten features:

- Current PM2.5 and PM10
- Temperature, relative humidity, and wind speed
- Hour and day of week
- PM2.5 values from one and three hours earlier
- Six-hour PM2.5 rolling mean

The target is `1` when the following hour's PM2.5 is above 35 µg/m³ and `0` otherwise.

Training compares a baseline, Logistic Regression, Random Forest, and Decision Tree. It uses a chronological 85/15 development-test split and five-fold time-series validation. The best non-baseline model is selected by validation F1 score.

The current saved model is a **Random Forest** with a decision threshold of **0.55**.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run from the project directory:

```bash
uv sync --dev
```

> `train.py` imports `python-dotenv`, which is not directly declared in `pyproject.toml`. If it is missing, run `uv add python-dotenv`.

## Run the data and training pipeline

Collect approximately three years of hourly data for Riyadh:

```bash
uv run python -m src.air_quality.collect
```

Build the processed dataset:

```bash
uv run python -m src.air_quality.features
```

Start MLflow in a separate terminal:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Train and evaluate the models:

```bash
uv run python -m src.air_quality.train
```

MLflow is available at `http://localhost:5000`. Training saves the selected model to `models/model.joblib` and evaluation reports to `reports/evaluation/`.

## Run the application

Start the API:

```bash
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start Streamlit in another terminal:

```bash
uv run streamlit run app/frontend/app.py
```

Open:

- Streamlit: `http://localhost:8501`
- Swagger API documentation: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

## API

Available endpoints:

- `GET /health` — service health
- `GET /model-info` — model name, threshold, features, and target
- `POST /predict` — next-hour pollution prediction

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pm2_5": 25.0,
    "pm10": 40.0,
    "temperature_2m": 35.0,
    "relative_humidity_2m": 30.0,
    "wind_speed_10m": 10.0,
    "hour": 14,
    "day_of_week": 2,
    "pm2_5_lag_1": 24.0,
    "pm2_5_lag_3": 23.0,
    "pm2_5_rolling_mean_6": 24.0
  }'
```

Example response:

```json
{
  "prediction": 1,
  "prediction_probability": 0.9876,
  "threshold": 0.55,
  "model": "Random Forest"
}
```

Successful predictions are appended to `data/predictions/predictions.csv`.

## Docker

Start the API, Streamlit, and MLflow services:

```bash
docker compose up --build
```

Stop them with:

```bash
docker compose down
```

## Tests

Run all API tests:

```bash
uv run pytest tests app/api/tests -q
```

Run lint checks:

```bash
uv run ruff check .
```

## Conclusion

This project demonstrates a complete MLOps workflow, from collecting and processing air-quality data to training, evaluating, tracking, and deploying a machine learning model. It provides a practical foundation for building reliable air-quality prediction systems and can be extended with automated monitoring and retraining.
