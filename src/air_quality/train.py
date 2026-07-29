from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from air_quality.features import FEATURES


TARGET = "high_pollution_next_hour"
DATA_PATH = "data/processed/model_table.parquet"
RANDOM_STATE = 42
SERIALIZATION = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
}


def split_by_time(df):
    df = df.sort_values("time") if "time" in df.columns else df.sort_index()
    df = df.reset_index(drop=True)

    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:validation_end].copy(),
        df.iloc[validation_end:].copy(),
    )


def make_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, FEATURES),
    ])

    return Pipeline([
        ("preprocess", preprocessing),
        ("model", model),
    ])


def best_threshold(y_true, probabilities):
    thresholds = np.arange(0.05, 0.91, 0.05)

    scores = [
        f1_score(
            y_true,
            probabilities >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ]

    return float(thresholds[np.argmax(scores)])


def calculate_metrics(y_true, probabilities, threshold, prefix):
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        f"{prefix}_accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        f"{prefix}_precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    if pd.Series(y_true).nunique() == 2:
        metrics[f"{prefix}_roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

    return metrics


def print_metrics(metrics, prefix):
    print(f"\n{prefix.replace('_', ' ').title()} Metrics:")

    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        key = f"{prefix}_{name}"

        if key in metrics:
            print(
                f"{name.replace('_', ' ').upper()}: "
                f"{metrics[key]:.4f}"
            )


def log_candidate(name, model, pipeline, threshold, metrics):
    display_name = DISPLAY_NAMES.get(
        name,
        name.replace("_", " ").title(),
    )

    parameter_names = {
        "strategy",
        "max_iter",
        "class_weight",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
    }

    parameters = {
        key: value
        for key, value in model.get_params().items()
        if key in parameter_names
    }

    mlflow_metrics = {
        key.replace("validation_", ""): value
        for key, value in metrics.items()
    }

    with mlflow.start_run(run_name=display_name):
        mlflow.log_params({
            "model": display_name,
            "threshold": threshold,
            "random_state": RANDOM_STATE,
            **parameters,
        })

        mlflow.log_metrics(mlflow_metrics)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=SERIALIZATION,
        )


def create_evaluation_reports(
    test,
    y_test,
    test_probabilities,
    test_predictions,
):
    evaluation_dir = Path("reports/evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    confusion_df = pd.DataFrame(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        ),
        index=[
            "actual_normal",
            "actual_high_pollution",
        ],
        columns=[
            "predicted_normal",
            "predicted_high_pollution",
        ],
    )

    confusion_path = evaluation_dir / "confusion_matrix.csv"
    confusion_df.to_csv(confusion_path, index=True)

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=[
                "normal",
                "high_pollution",
            ],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    report_path = evaluation_dir / "classification_report.csv"
    report_df.to_csv(report_path, index=True)

    error_analysis = test[FEATURES + [TARGET]].copy()
    error_analysis["prediction_probability"] = test_probabilities
    error_analysis["prediction"] = test_predictions
    error_analysis["correct"] = (
        error_analysis[TARGET]
        == error_analysis["prediction"]
    )
    error_analysis["error_type"] = "correct"

    false_positive = (
        (error_analysis[TARGET] == 0)
        & (error_analysis["prediction"] == 1)
    )

    false_negative = (
        (error_analysis[TARGET] == 1)
        & (error_analysis["prediction"] == 0)
    )

    error_analysis.loc[
        false_positive,
        "error_type",
    ] = "false_positive"

    error_analysis.loc[
        false_negative,
        "error_type",
    ] = "false_negative"

    errors_path = evaluation_dir / "error_analysis.csv"
    error_analysis.to_csv(errors_path, index=False)

    return {
        "confusion_df": confusion_df,
        "report_df": report_df,
        "error_analysis": error_analysis,
        "confusion_path": confusion_path,
        "report_path": report_path,
        "errors_path": errors_path,
    }


def main():
    df = pd.read_parquet(DATA_PATH)

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    train, validation, test = split_by_time(df)

    for name, dataset in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        print(f"\n{name} shape:", dataset.shape)
        print(f"{name} distribution:")
        print(dataset[TARGET].value_counts().sort_index())

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype(int)

    X_test = test[FEATURES]
    y_test = test[TARGET].astype(int)

    models = {
        "baseline": DummyClassifier(
            strategy="most_frequent",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("riyadh-air-quality")
    mlflow.sklearn.autolog(disable=True)

    results = []

    best = {
        "name": None,
        "pipeline": None,
        "threshold": 0.50,
        "f1": -1.0,
    }

    for name, model in models.items():
        display_name = DISPLAY_NAMES.get(
            name,
            name.replace("_", " ").title(),
        )

        print(f"\nTraining: {display_name}")

        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        threshold = best_threshold(
            y_validation,
            probabilities,
        )

        metrics = calculate_metrics(
            y_validation,
            probabilities,
            threshold,
            "validation",
        )

        results.append({
            "model": display_name,
            "threshold": threshold,
            **metrics,
        })

        log_candidate(
            name,
            model,
            pipeline,
            threshold,
            metrics,
        )

        print("Best threshold:", round(threshold, 2))
        print_metrics(metrics, "validation")

        if metrics["validation_f1"] > best["f1"]:
            best = {
                "name": display_name,
                "pipeline": pipeline,
                "threshold": threshold,
                "f1": metrics["validation_f1"],
            }

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "validation_f1",
        ascending=False,
    )

    print("\nValidation Results:")
    print(results_df.to_string(index=False))

    if best["pipeline"] is None:
        raise RuntimeError("No model was selected.")

    test_probabilities = best["pipeline"].predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best["threshold"]
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best["threshold"],
        "test",
    )

    print("\nFinal Test Results:")
    print("Selected model:", best["name"])
    print(
        "Selected threshold:",
        round(best["threshold"], 2),
    )
    print_metrics(test_metrics, "test")

    reports = create_evaluation_reports(
        test,
        y_test,
        test_probabilities,
        test_predictions,
    )

    print("\nConfusion Matrix:")
    print(reports["confusion_df"])

    print("\nClassification Report:")
    print(reports["report_df"])

    print("\nError Analysis Counts:")
    print(
        reports["error_analysis"]
        ["error_type"]
        .value_counts()
    )

    with mlflow.start_run(
        run_name=f"Selected Model Test - {best['name']}"
    ):
        mlflow.log_params({
            "selected_model": best["name"],
            "selected_threshold": best["threshold"],
        })

        simple_test_metrics = {
            key.replace("test_", ""): value
            for key, value in test_metrics.items()
        }

        mlflow.log_metrics(simple_test_metrics)

        for path_key in (
            "confusion_path",
            "report_path",
            "errors_path",
        ):
            mlflow.log_artifact(
                str(reports[path_key]),
                artifact_path="evaluation",
            )

        mlflow.sklearn.log_model(
            best["pipeline"],
            name="selected_model",
            serialization_format=SERIALIZATION,
        )

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "threshold": best["threshold"],
            "model_name": best["name"],
            "features": FEATURES,
            "target": TARGET,
        },
        "models/model.joblib",
    )

    monitoring_dir = Path("data/monitoring")

    monitoring_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = test[FEATURES + [TARGET]].copy()
    reference["prediction_probability"] = test_probabilities
    reference["prediction"] = test_predictions

    reference.to_csv(
        monitoring_dir / "reference.csv",
        index=False,
    )

    print("\nSaved files:")

    for path in (
        "models/model.joblib",
        "data/monitoring/reference.csv",
        "reports/evaluation/confusion_matrix.csv",
        "reports/evaluation/classification_report.csv",
        "reports/evaluation/error_analysis.csv",
    ):
        print(path)


if __name__ == "__main__":
    main()