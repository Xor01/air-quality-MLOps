# Riyadh Air Quality MLOps

An end-to-end machine learning project that predicts whether **PM2.5 pollution in Riyadh will be high during the next hour**. The repository covers data collection, feature engineering, time-aware model evaluation, experiment tracking, model serving, a web interface, prediction logging, testing, and containerized deployment.

The binary target is `high_pollution_next_hour`: a value of `1` means the next hour's PM2.5 concentration is expected to exceed **35 µg/m³**, while `0` means it is expected to remain at or below that threshold.


## Project overview

This project uses historical air-quality and weather observations for central Riyadh (`24.7136`, `46.6753`). It combines data from two Open-Meteo endpoints:

- Air Quality API: hourly PM2.5 and PM10 concentrations.
- Historical Weather API: hourly temperature, relative humidity, and wind speed.

The system then:

1. Downloads approximately three years of hourly observations.
2. Merges air-quality and weather records by timestamp.
3. Creates calendar, lag, and rolling-window features.
4. Builds a next-hour high-pollution classification target.
5. Compares several classifiers using time-series cross-validation.
6. Selects the non-baseline model with the best validation F1 score.
7. Evaluates it once on a held-out final time period.
8. Saves the model bundle, evaluation reports, and monitoring reference data.
9. Serves predictions through FastAPI and a Streamlit user interface.

The repository currently includes a trained **Random Forest** bundle with a decision threshold of **0.55**. These values may change after retraining.

## Architecture and workflow

```text
Open-Meteo APIs
      |
      v
data/raw/air_quality.json
      |
      v
Feature engineering
      |
      v
data/processed/model_table.parquet
      |
      v
Time-aware training + MLflow tracking
      |
      +--> models/model.joblib
      +--> reports/evaluation/*.csv
      +--> data/monitoring/reference.csv
                 |
                 v
          FastAPI prediction API <---- Streamlit UI
                 |
                 v
        data/predictions/predictions.csv
```

## Technology stack

- **Python 3.10+**; the repository pins Python 3.13 in `.python-version`.
- **uv** for dependency and virtual-environment management.
- **pandas** and **PyArrow** for data preparation and Parquet storage.
- **scikit-learn** for preprocessing, training, and evaluation.
- **MLflow** for experiment, metric, model, and artifact tracking.
- **FastAPI** and **Pydantic** for the prediction service and request validation.
- **Streamlit** for the browser-based prediction interface.
- **pytest**, **pytest-cov**, and **Ruff** for development checks.
- **Docker Compose** for running the API, frontend, and MLflow services.

## Repository structure

```text
air-quality-mlops/
├── app/
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   └── tests/                  # Additional API tests (not in default test path)
│   └── frontend/
│       └── app.py                  # Streamlit application
├── data/
│   ├── raw/air_quality.json        # Combined Open-Meteo responses
│   ├── processed/model_table.parquet
│   ├── monitoring/reference.csv    # Held-out reference observations
│   └── predictions/predictions.csv # Append-only inference log
├── mlartifacts/                    # Locally generated MLflow artifacts
├── models/model.joblib             # Deployable model bundle
├── reports/evaluation/             # Final test reports and error analysis
├── src/air_quality/
│   ├── collect.py                  # Data acquisition
│   ├── features.py                 # Dataset and target construction
│   ├── train.py                    # Training, selection, and evaluation
│   └── monitoring.py               # Reserved for future monitoring logic
├── tests/test_api.py               # Default pytest suite
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── pyproject.toml
└── uv.lock
```

`main.py` is the default placeholder created with the package, and `test.py` is a small dataset-inspection script. The production entry points are the modules under `src/air_quality/` and `app/`.

## Data and features

### Raw data

`src/air_quality/collect.py` requests data ending five days before the current date and going back 1,095 days. The five-day delay accommodates availability in the historical weather endpoint. Dates and coordinates are stored alongside both API responses in `data/raw/air_quality.json`.

The checked-in dataset contains **26,247 processed hourly rows**, covering `2023-07-26 06:00` through `2026-07-25 23:00`. It contains 18,949 positive and 7,298 negative target examples. These figures describe the current artifact and will change when data is recollected.

### Input features

| Feature | Description |
|---|---|
| `pm2_5` | Current PM2.5 concentration |
| `pm10` | Current PM10 concentration |
| `temperature_2m` | Air temperature at 2 metres |
| `relative_humidity_2m` | Relative humidity at 2 metres, from 0 to 100 |
| `wind_speed_10m` | Wind speed at 10 metres |
| `hour` | Hour of day, from 0 to 23 |
| `day_of_week` | Day index where Monday is 0 and Sunday is 6 |
| `pm2_5_lag_1` | PM2.5 value one hour earlier |
| `pm2_5_lag_3` | PM2.5 value three hours earlier |
| `pm2_5_rolling_mean_6` | Mean PM2.5 over the preceding six hours |

### Target construction

The target is calculated as:

```text
high_pollution_next_hour = 1 if next_hour_pm2_5 > 35.0 else 0
```

Rows lacking the required lag, rolling, or shifted-target values are removed. The default `35.0` threshold can be changed through the `threshold` argument of `build_dataset()` in `src/air_quality/features.py`.

## Model training and evaluation

The training pipeline evaluates four candidates:

- Most-frequent `DummyClassifier` as a baseline.
- Class-balanced Logistic Regression.
- Class-balanced Random Forest.
- Class-balanced Decision Tree.

Each model is wrapped in a scikit-learn pipeline that:

1. Selects the ten numeric features.
2. Imputes missing values with the median.
3. Standardizes the features.
4. Fits the classifier.

Evaluation is designed for temporal data:

- The latest 15% of observations are held out as a final test set.
- The earlier 85% are evaluated using five-fold `TimeSeriesSplit` validation.
- Folds without both target classes are skipped.
- A probability threshold from 0.05 through 0.90 is selected by validation F1 score.
- The baseline is logged and reported but cannot be selected as the final model.
- Accuracy, precision, recall, F1, and ROC AUC are recorded when applicable.

The current checked-in final evaluation reports show:

| Metric | Value |
|---|---:|
| Accuracy | 0.9893 |
| High-pollution precision | 0.9966 |
| High-pollution recall | 0.9925 |
| High-pollution F1 | 0.9945 |
| Test observations | 3,938 |

The final confusion matrix contains 78 true negatives, 13 false positives, 29 false negatives, and 3,818 true positives. Because the target is imbalanced toward high-pollution observations, review the class-specific results and error analysis rather than accuracy alone.

## Getting started

### Prerequisites

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) for the documented local workflow.
- Docker and Docker Compose if you want to run the containerized stack.
- Internet access only when collecting new data or pulling container images.

### Install dependencies

From the repository directory:

```bash
uv sync --dev
```

This creates or updates `.venv` from `pyproject.toml` and the locked dependency versions in `uv.lock`.

> Note: `src/air_quality/train.py` imports `python-dotenv`, but it is not currently declared directly in `pyproject.toml`. It may be available transitively in the existing environment. If a fresh environment reports `ModuleNotFoundError: dotenv`, add it explicitly with `uv add python-dotenv`.

## Running the complete pipeline

Run commands from the `air-quality-mlops` directory so that all relative paths resolve correctly.

### 1. Collect historical data

```bash
uv run python -m src.air_quality.collect
```

Output: `data/raw/air_quality.json`

### 2. Build the model table

```bash
uv run python -m src.air_quality.features
```

Output: `data/processed/model_table.parquet`

You can inspect the processed table with:

```bash
uv run python test.py
```

### 3. Start MLflow

Training defaults to `http://localhost:5000`, so start a tracking server in a separate terminal:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Open `http://localhost:5000` to inspect runs.

To use another tracking server, define `TRACKING_URL` before training. For example, in PowerShell:

```powershell
$env:TRACKING_URL = "http://localhost:5000"
```

### 4. Train and evaluate

```bash
uv run python -m src.air_quality.train
```

Training writes the selected deployable bundle to `models/model.joblib`, creates CSV reports under `reports/evaluation/`, creates `data/monitoring/reference.csv`, and logs each candidate and the selected test run to MLflow.

## Running the applications

The API loads `models/model.joblib` at import time. Ensure that this file exists before starting the service.

### FastAPI service

```bash
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful local URLs:

- API root: `http://localhost:8000`
- Interactive Swagger documentation: `http://localhost:8000/docs`
- Alternative ReDoc documentation: `http://localhost:8000/redoc`
- Health endpoint: `http://localhost:8000/health`

### Streamlit frontend

With the API running on port 8000, open another terminal and run:

```bash
uv run streamlit run app/frontend/app.py
```

The frontend defaults to `http://127.0.0.1:8000`. Override the API location with the `API_URL` environment variable when needed.

## API reference

### `GET /health`

Reports whether the service and model are loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `GET /model-info`

Returns the selected model name, probability threshold, ordered feature list, target name, and API version.

### `POST /predict`

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

Input validation enforces non-negative pollutant, humidity-lag, rolling, and wind values; humidity from 0 to 100; hour from 0 to 23; and day of week from 0 to 6. Invalid payloads return HTTP `422`.

Every successful prediction is appended to `data/predictions/predictions.csv` with a UUID request ID, UTC timestamp, inputs, probability, and predicted class. Concurrent production writes would require a more robust storage mechanism than a shared CSV file.

## Docker deployment

Build and start the complete stack:

```bash
docker compose up --build
```

The Compose file exposes:

| Service | Host port | Purpose |
|---|---:|---|
| `api` | 8000 | FastAPI prediction service |
| `frontend` | 8501 | Streamlit interface |
| `mlflow` | 5000 | MLflow tracking UI and server |

The frontend waits for the API health check and communicates with it at `http://api:8000`. The API mounts the local `models` directory read-only and uses a named volume for monitoring data. MLflow stores its SQLite database and artifacts in a named volume.

The Compose file also contains deployment-oriented DigitalOcean Container Registry image names. Local `--build` usage builds those services from `Dockerfile.api` and `Dockerfile.frontend`; update the image names for your own registry before publishing.

Stop the stack with:

```bash
docker compose down
```

Add `-v` only when you intentionally want to remove the named monitoring and MLflow volumes as well.

## Testing and code quality

Run the configured test suite:

```bash
uv run pytest -q
```

The default pytest configuration points to `tests/`. The additional tests under `app/api/tests/` are not discovered by that configuration unless their path is supplied explicitly:

```bash
uv run pytest tests app/api/tests -q
```

Run coverage:

```bash
uv run pytest --cov=app --cov=src --cov-report=term-missing
```

Run Ruff checks:

```bash
uv run ruff check .
```

The API tests load the model and successful prediction tests append rows to `data/predictions/predictions.csv`. Expect that tracked file to change after running them.

## Generated artifacts

| Path | Contents |
|---|---|
| `models/model.joblib` | Dictionary containing the fitted pipeline, threshold, model name, features, and target |
| `reports/evaluation/confusion_matrix.csv` | Final test confusion matrix |
| `reports/evaluation/classification_report.csv` | Precision, recall, F1, and support by class |
| `reports/evaluation/error_analysis.csv` | Inputs, actual labels, probabilities, predictions, and error types |
| `data/monitoring/reference.csv` | Final test slice enriched with predictions for future monitoring |
| `data/predictions/predictions.csv` | Online inference log |
| `mlflow.db` | Local MLflow SQLite backend store |
| `mlartifacts/` | Models and evaluation artifacts logged by MLflow |

Many of these files are regenerated or appended to during normal operation. Avoid committing incidental changes unless updated artifacts are intentionally part of the revision.

## Monitoring and operational notes

The training process creates a reference dataset suitable for drift or quality monitoring, and the API records inference traffic. The project declares Evidently as a dependency, but `src/air_quality/monitoring.py` is currently empty; automated drift reports, alert thresholds, and scheduled monitoring are not yet implemented.

For production use, consider adding:

- Schema-controlled storage for prediction events.
- Observed-label collection for delayed performance evaluation.
- Data and prediction drift reports based on `reference.csv`.
- Alerting for invalid input distributions, service errors, and drift.
- Model versioning and an explicit promotion or rollback workflow.
- Authentication, rate limiting, request tracing, and structured logs.

## Known limitations

- The model represents one fixed Riyadh coordinate, not the entire city or other regions.
- Predictions require lag and rolling features to be supplied by the caller; the API does not fetch or maintain recent sensor history.
- The PM2.5 target threshold is a project configuration and should not be interpreted as medical advice or an official air-quality warning.
- There is no scheduled ingestion, retraining, deployment, or monitoring workflow.
- The model is loaded only at API startup; replacing the file requires restarting the service.
- Prediction logging uses an append-only local CSV and is not designed for concurrent multi-worker writes.
- The current dataset is class-imbalanced, so evaluation should include per-class metrics and error costs.
- `python-dotenv` should be declared directly before relying on clean-environment training.
- No license is currently included in the repository. Add one before redistributing the project.
