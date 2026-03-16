from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "satisfaction"
TARGET_MAP = {"neutral or dissatisfied": 0, "satisfied": 1}
IDENTIFIER_COLUMNS = ["row_id", "Unnamed: 0", "id"]
DEFAULT_OUTPUT_DIR = Path("outputs") / "debug_model"


def resolve_input_path(preferred: str, fallback: str) -> Path:
    preferred_path = Path(preferred)
    if preferred_path.exists():
        return preferred_path

    fallback_path = Path(fallback)
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"Neither {preferred!r} nor {fallback!r} exists.")


def load_combined_data() -> tuple[pd.DataFrame, dict[str, str]]:
    train_path = resolve_input_path("data/train.csv", "train.csv")
    test_path = resolve_input_path("data/test.csv", "test.csv")

    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)
    combined = pd.concat([raw_train, raw_test], axis=0, ignore_index=True).copy()
    combined["row_id"] = range(len(combined))

    observed_labels = set(combined[TARGET].dropna().unique())
    unexpected_labels = sorted(observed_labels.difference(TARGET_MAP))
    if unexpected_labels:
        raise ValueError(f"Unexpected target labels found: {unexpected_labels}")

    combined[TARGET] = combined[TARGET].map(TARGET_MAP).astype("int64")
    return combined, {"train": str(train_path), "test": str(test_path)}


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET],
        random_state=SEED,
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET],
        random_state=SEED,
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_split(
    estimator: Pipeline,
    split_name: str,
    split_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    X_split = split_df[feature_columns].copy()
    y_true = split_df[TARGET].astype(int)
    y_prob = estimator.predict_proba(X_split)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "rows": int(len(split_df)),
    }

    confusion_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )

    prediction_df = split_df[["row_id", TARGET]].copy().rename(columns={TARGET: "y_true"})
    if "id" in split_df.columns:
        prediction_df["id"] = split_df["id"]
    prediction_df["y_prob"] = y_prob
    prediction_df["y_pred"] = y_pred

    return metrics, confusion_df, prediction_df


def build_feature_manifest(
    feature_columns: list[str], numeric_features: list[str], categorical_features: list[str]
) -> dict[str, object]:
    return {
        "target": TARGET,
        "included_features": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_features": {
            "row_id": "generated audit identifier",
            "Unnamed: 0": "identifier-like source index column excluded from modelling",
            "id": "identifier-like passenger id column excluded from modelling",
        },
    }


def build_split_manifest(
    source_paths: dict[str, str],
    combined: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "status": "CONFIRMED BY EXECUTION",
        "seed": SEED,
        "source_paths": source_paths,
        "split_rule": "custom stratified 70/15/15 split built from combined original CSV files",
        "target": TARGET,
        "class_encoding": TARGET_MAP,
        "combined_shape": list(combined.shape),
        "train_shape": list(train_df.shape),
        "validation_shape": list(validation_df.shape),
        "test_shape": list(test_df.shape),
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(output_dir: Path | str = DEFAULT_OUTPUT_DIR, write_outputs: bool = True) -> dict[str, object]:
    output_dir = Path(output_dir)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    combined, source_paths = load_combined_data()
    train_df, validation_df, test_df = split_data(combined)

    feature_columns = [column for column in train_df.columns if column not in IDENTIFIER_COLUMNS + [TARGET]]
    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET].astype(int)

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    estimator = build_pipeline(numeric_features, categorical_features)
    estimator.fit(X_train, y_train)

    threshold = 0.5
    validation_metrics, validation_confusion, validation_predictions = evaluate_split(
        estimator, "validation", validation_df, feature_columns, threshold
    )
    test_metrics, test_confusion, test_predictions = evaluate_split(
        estimator, "test", test_df, feature_columns, threshold
    )

    feature_manifest = build_feature_manifest(feature_columns, numeric_features, categorical_features)
    split_manifest = build_split_manifest(source_paths, combined, train_df, validation_df, test_df)
    encoded_feature_count = int(estimator.named_steps["preprocessor"].get_feature_names_out().shape[0])
    run_metadata = {
        "status": "CONFIRMED BY EXECUTION",
        "task_label": "DEBUG_BROKEN_PIPELINE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "threshold": threshold,
        "model_type": "LogisticRegression",
        "feature_count_before_encoding": len(feature_columns),
        "feature_count_after_encoding": encoded_feature_count,
        "source_paths": source_paths,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    if write_outputs:
        dump(estimator, output_dir / "model.joblib")
        save_json(output_dir / "metrics_validation.json", validation_metrics)
        save_json(output_dir / "metrics_test.json", test_metrics)
        validation_confusion.to_csv(output_dir / "confusion_matrix_validation.csv")
        test_confusion.to_csv(output_dir / "confusion_matrix_test.csv")
        validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
        save_json(output_dir / "feature_manifest.json", feature_manifest)
        save_json(output_dir / "split_manifest.json", split_manifest)
        save_json(output_dir / "run_metadata.json", run_metadata)
        split_assignments = pd.concat(
            [
                train_df[["row_id"]].assign(split="train"),
                validation_df[["row_id"]].assign(split="validation"),
                test_df[["row_id"]].assign(split="test"),
            ],
            ignore_index=True,
        )
        split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
        run_log_lines = [
            f"task_label=DEBUG_BROKEN_PIPELINE",
            f"seed={SEED}",
            f"threshold={threshold}",
            f"train_rows={len(train_df)}",
            f"validation_rows={len(validation_df)}",
            f"test_rows={len(test_df)}",
            f"validation_roc_auc={validation_metrics['roc_auc']:.10f}",
            f"validation_pr_auc={validation_metrics['pr_auc']:.10f}",
            f"validation_accuracy={validation_metrics['accuracy']:.10f}",
            f"validation_precision={validation_metrics['precision']:.10f}",
            f"validation_recall={validation_metrics['recall']:.10f}",
            f"validation_f1_score={validation_metrics['f1_score']:.10f}",
            f"test_roc_auc={test_metrics['roc_auc']:.10f}",
            f"test_pr_auc={test_metrics['pr_auc']:.10f}",
            f"test_accuracy={test_metrics['accuracy']:.10f}",
            f"test_precision={test_metrics['precision']:.10f}",
            f"test_recall={test_metrics['recall']:.10f}",
            f"test_f1_score={test_metrics['f1_score']:.10f}",
        ]
        (output_dir / "run_log.txt").write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")

    return {
        "estimator": estimator,
        "output_dir": str(output_dir),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_confusion": validation_confusion,
        "test_confusion": test_confusion,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "feature_manifest": feature_manifest,
        "split_manifest": split_manifest,
        "run_metadata": run_metadata,
    }


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Corrected baseline pipeline for Airline Passenger Satisfaction.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where model artefacts will be written.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    results = run_pipeline(output_dir=args.output_dir, write_outputs=True)
    printable = {
        "status": "CONFIRMED BY EXECUTION",
        "output_dir": results["output_dir"],
        "validation_metrics": results["validation_metrics"],
        "test_metrics": results["test_metrics"],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
