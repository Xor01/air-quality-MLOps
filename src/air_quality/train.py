from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from air_quality.features import FEATURES


TARGET = "high_pollution_next_hour"
DATA_PATH = "data/processed/model_table.parquet"
RANDOM_STATE = 42
TEST_SIZE = 0.15
N_TIME_SPLITS = 5
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

    test_start = int(len(df) * (1 - TEST_SIZE))
    development = df.iloc[:test_start].copy()
    test = df.iloc[test_start:].copy()

    if development.empty or test.empty:
        raise ValueError("The dataset is too small for a temporal split.")

    return development, test


def require_two_classes(y, name):
    classes = pd.Series(y).dropna().unique()

    if len(classes) < 2:
        raise ValueError(
            f"{name} contains only one class: {classes.tolist()}. "
            "Use more historical data or revise the target definition."
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


def positive_probability(pipeline, X):
    probabilities = pipeline.predict_proba(X)
    classes = pipeline.named_steps["model"].classes_
    class_one_index = np.where(classes == 1)[0]

    if len(class_one_index) != 1:
        raise ValueError("The trained model must contain classes 0 and 1.")

    return probabilities[:, class_one_index[0]]


def best_threshold(y_true, probabilities):
    require_two_classes(y_true, "Validation data")
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
            precision_score(y_true, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        f"{prefix}_f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
    }

    if pd.Series(y_true).nunique() == 2:
        metrics[f"{prefix}_roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

    return metrics


def print_metrics(metrics, prefix):
    print(f"\n{prefix.replace('_', ' ').title()} Metrics:")

    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        key = f"{prefix}_{name}"

        if key in metrics:
            print(f"{name.replace('_', ' ').upper()}: {metrics[key]:.4f}")


def temporal_validation(model, development):
    splitter = TimeSeriesSplit(n_splits=N_TIME_SPLITS)
    all_targets = []
    all_probabilities = []
    valid_folds = 0

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(development),
        start=1,
    ):
        fold_train = development.iloc[train_indices]
        fold_validation = development.iloc[validation_indices]

        X_train = fold_train[FEATURES]
        y_train = fold_train[TARGET].astype(int)
        X_validation = fold_validation[FEATURES]
        y_validation = fold_validation[TARGET].astype(int)

        print(f"\nFold {fold} train distribution:")
        print(y_train.value_counts().sort_index())

        print(f"Fold {fold} validation distribution:")
        print(y_validation.value_counts().sort_index())

        if y_train.nunique() < 2 or y_validation.nunique() < 2:
            print(f"Skipping fold {fold}: it does not contain both classes.")
            continue

        pipeline = make_pipeline(clone(model))
        pipeline.fit(X_train, y_train)

        probabilities = positive_probability(pipeline, X_validation)

        all_targets.append(y_validation.to_numpy())
        all_probabilities.append(probabilities)
        valid_folds += 1

    if valid_folds == 0:
        raise ValueError(
            "No temporal validation fold contains both classes. "
            "Use more historical data, reduce N_TIME_SPLITS, "
            "or revise the target definition."
        )

    return (
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
        valid_folds,
    )


def log_candidate(name, model, pipeline, threshold, metrics, valid_folds):
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
            "valid_time_folds": valid_folds,
            "eligible_for_selection": name != "baseline",
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
        error_analysis[TARGET] == error_analysis["prediction"]
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

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    require_two_classes(df[TARGET], "Complete dataset")

    development, test = split_by_time(df)

    require_two_classes(
        development[TARGET],
        "Development data",
    )

    require_two_classes(
        test[TARGET],
        "Final test data",
    )

    print("\nDevelopment distribution:")
    print(development[TARGET].value_counts().sort_index())

    print("\nTest distribution:")
    print(test[TARGET].value_counts().sort_index())

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

        print(f"\nEvaluating: {display_name}")

        (
            validation_targets,
            validation_probabilities,
            valid_folds,
        ) = temporal_validation(
            model,
            development,
        )

        threshold = best_threshold(
            validation_targets,
            validation_probabilities,
        )

        metrics = calculate_metrics(
            validation_targets,
            validation_probabilities,
            threshold,
            "validation",
        )

        pipeline = make_pipeline(clone(model))

        pipeline.fit(
            development[FEATURES],
            development[TARGET].astype(int),
        )

        results.append({
            "model": display_name,
            "valid_time_folds": valid_folds,
            "threshold": threshold,
            "eligible_for_selection": name != "baseline",
            **metrics,
        })

        log_candidate(
            name,
            model,
            pipeline,
            threshold,
            metrics,
            valid_folds,
        )

        print("Best threshold:", round(threshold, 2))
        print_metrics(metrics, "validation")

        if (
            name != "baseline"
            and metrics["validation_f1"] > best["f1"]
        ):
            best = {
                "name": display_name,
                "pipeline": pipeline,
                "threshold": threshold,
                "f1": metrics["validation_f1"],
            }

    results_df = pd.DataFrame(results).sort_values(
        "validation_f1",
        ascending=False,
    )

    print("\nTemporal Validation Results:")
    print(results_df.to_string(index=False))

    if best["pipeline"] is None:
        raise RuntimeError("No trainable model was selected.")

    X_test = test[FEATURES]
    y_test = test[TARGET].astype(int)

    test_probabilities = positive_probability(
        best["pipeline"],
        X_test,
    )

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
    print("Selected threshold:", round(best["threshold"], 2))
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
    monitoring_dir.mkdir(parents=True, exist_ok=True)

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