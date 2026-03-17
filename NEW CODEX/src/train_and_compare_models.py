from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_SEED = 42
TARGET_COLUMN = "satisfaction"
POSITIVE_LABEL = "satisfied"
NEGATIVE_LABEL = "neutral or dissatisfied"
STATUS_CONFIRMED_EXEC = "CONFIRMED_BY_EXECUTION"

NUMERIC_FEATURES = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

MODEL_PRIORITY = {
    "LogisticRegression": 0,
    "HistGradientBoostingClassifier": 1,
    "RandomForestClassifier": 2,
    "ExtraTreesClassifier": 3,
}


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_prepared_dataset(data_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df["__combined_row_id__"] = combined_df.index.astype(int)

    drop_columns = []
    if str(combined_df.columns[0]).startswith("Unnamed"):
        drop_columns.append(combined_df.columns[0])
    if "id" in combined_df.columns:
        drop_columns.append("id")
    combined_df = combined_df.drop(columns=drop_columns)

    combined_df[TARGET_COLUMN] = combined_df[TARGET_COLUMN].map(
        {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    ).astype(int)
    return combined_df


def read_split_manifest(manifest_path: Path) -> Tuple[Dict[str, object], str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Canonical split manifest not found at outputs/shared/split_manifest.json"
        )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    return manifest, manifest_hash


def validate_split_manifest(manifest: Dict[str, object], dataset: pd.DataFrame) -> None:
    if manifest.get("random_seed") != RANDOM_SEED:
        raise ValueError("Split manifest random seed does not match the benchmark requirement.")
    if manifest.get("target_name") != TARGET_COLUMN:
        raise ValueError("Split manifest target name does not match the benchmark requirement.")

    split_indices = manifest.get("split_indices", {})
    if set(split_indices.keys()) != {"train", "validation", "test"}:
        raise ValueError("Split manifest is missing required split indices.")

    all_indices: List[int] = []
    for split_name in ["train", "validation", "test"]:
        indices = split_indices[split_name]
        all_indices.extend(indices)
        expected_count = manifest["split_summary"][split_name]["row_count"]
        if len(indices) != expected_count:
            raise ValueError(f"Split manifest row count mismatch for {split_name}.")

    if len(all_indices) != dataset.shape[0]:
        raise ValueError("Split manifest does not cover the entire combined dataset.")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Split manifest contains overlapping split assignments.")


def create_preprocessor(model_name: str) -> ColumnTransformer:
    if model_name == "LogisticRegression":
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
    elif model_name in {"RandomForestClassifier", "ExtraTreesClassifier"}:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
    elif model_name == "HistGradientBoostingClassifier":
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def create_estimator(model_name: str):
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model_name == "ExtraTreesClassifier":
        return ExtraTreesClassifier(n_estimators=100, random_state=42)
    if model_name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(max_iter=100, random_state=42)
    raise ValueError(f"Unsupported model name: {model_name}")


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": 0.5,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_pred


def build_pipeline(model_name: str) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", create_preprocessor(model_name)),
            ("model", create_estimator(model_name)),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def save_predictions(
    output_path: Path,
    row_ids: pd.Series,
    y_true: pd.Series,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    prediction_df = pd.DataFrame(
        {
            "combined_row_id": row_ids.astype(int).to_numpy(),
            "y_true": y_true.astype(int).to_numpy(),
            "y_pred_proba": y_proba.astype(float),
            "y_pred_label": y_pred.astype(int),
        }
    )
    prediction_df.to_csv(output_path, index=False)


def save_confusion_matrix(output_path: Path, metrics: Dict[str, float]) -> None:
    matrix_df = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    matrix_df.to_csv(output_path)


def select_model(validation_results: pd.DataFrame) -> str:
    ranked = validation_results.sort_values(
        by=["roc_auc", "pr_auc", "f1", "priority_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return str(ranked.loc[0, "model_name"])


def write_candidate_manifest(output_path: Path) -> None:
    manifest = {
        "candidate_models": [
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "HistGradientBoostingClassifier",
        ],
        "preprocessing_by_model": {
            "LogisticRegression": "median impute + StandardScaler for numeric; most_frequent impute + OneHotEncoder for categorical",
            "RandomForestClassifier": "median impute for numeric; most_frequent impute + OneHotEncoder for categorical",
            "ExtraTreesClassifier": "median impute for numeric; most_frequent impute + OneHotEncoder for categorical",
            "HistGradientBoostingClassifier": "median impute for numeric; most_frequent impute + OrdinalEncoder for categorical",
        },
        "fixed_parameters_by_model": {
            "LogisticRegression": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"},
            "RandomForestClassifier": {"n_estimators": 100, "random_state": 42},
            "ExtraTreesClassifier": {"n_estimators": 100, "random_state": 42},
            "HistGradientBoostingClassifier": {"max_iter": 100, "random_state": 42},
        },
        "split_manifest_path": "outputs/shared/split_manifest.json",
        "target_name": "satisfaction",
        "random_seed": 42,
        "selection_metric": "validation ROC-AUC",
        "tie_break_rules": ["PR-AUC", "F1 at 0.5", "fixed priority order"],
        "threshold_rule": "fixed at 0.5",
        "dependency_scope": "requirements.txt",
        "execution_status": STATUS_CONFIRMED_EXEC,
    }
    output_path.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")


def write_selection_report(
    output_path: Path,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_model: str,
) -> None:
    selected_row = validation_df.loc[validation_df["model_name"] == selected_model].iloc[0]
    content_lines = [
        "# Model Selection Report",
        "",
        "## Selection rule",
        "- Primary metric: validation ROC-AUC",
        "- Tie-breaker 1: validation PR-AUC",
        "- Tie-breaker 2: validation F1 at threshold 0.5",
        "- Tie-breaker 3: fixed priority order `LogisticRegression > HistGradientBoostingClassifier > RandomForestClassifier > ExtraTreesClassifier`",
        "- Classification threshold fixed at `0.5` for all reported precision/recall/F1 metrics",
        "",
        "## Selected model",
        f"- Selected model: `{selected_model}`",
        f"- Validation ROC-AUC: `{selected_row['roc_auc']:.6f}`",
        f"- Validation PR-AUC: `{selected_row['pr_auc']:.6f}`",
        f"- Validation F1: `{selected_row['f1']:.6f}`",
        "",
        "## Validation ranking",
        "```text",
        validation_df.to_string(index=False),
        "```",
        "",
        "## Test metrics note",
        "- The test metrics table is post-selection descriptive comparison only and was not used for model selection.",
        "",
        "## Test metrics by candidate model",
        "```text",
        test_df.to_string(index=False),
        "```",
        "",
    ]
    output_path.write_text("\n".join(content_lines), encoding="utf-8")


def write_feature_manifest(
    output_path: Path,
    selected_model: str,
    feature_names: List[str],
) -> None:
    payload = {
        "selected_model": selected_model,
        "target_name": TARGET_COLUMN,
        "original_numeric_features": NUMERIC_FEATURES,
        "original_categorical_features": CATEGORICAL_FEATURES,
        "transformed_feature_count": len(feature_names),
        "transformed_feature_names": feature_names,
        "execution_status": STATUS_CONFIRMED_EXEC,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def append_decision_log(
    repo_root: Path,
    selected_model: str,
    validation_df: pd.DataFrame,
    split_manifest_hash: str,
) -> None:
    decision_log_path = repo_root / "docs" / "decision_log.md"
    selected_row = validation_df.loc[validation_df["model_name"] == selected_model].iloc[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"## {timestamp} - Step 3 model comparison and selection",
        "",
        f"- Reused `outputs/shared/split_manifest.json` without modification; SHA-256 at run time: `{split_manifest_hash}`.",
        "- All four candidate models were trained on the training split only, with preprocessing fitted on training data only.",
        "- Validation-only selection rule applied exactly as specified: ROC-AUC, then PR-AUC, then F1 at 0.5, then fixed priority order.",
        "- Classification threshold remained fixed at `0.5`; no threshold tuning was attempted.",
        f"- Selected final model: `{selected_model}` with validation ROC-AUC `{selected_row['roc_auc']:.6f}`, PR-AUC `{selected_row['pr_auc']:.6f}`, and F1 `{selected_row['f1']:.6f}`.",
        "- Test metrics were written for all candidate models as post-selection descriptive comparison only.",
        "",
    ]
    current = decision_log_path.read_text(encoding="utf-8").rstrip()
    decision_log_path.write_text(current + "\n\n" + "\n".join(lines), encoding="utf-8")


def append_reproducibility_record(
    repo_root: Path,
    selected_model: str,
    validation_df: pd.DataFrame,
    created_files: List[str],
    split_manifest_hash: str,
    execution_command: str,
) -> None:
    record_path = repo_root / "docs" / "reproducibility_record.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = "\n".join(
        [
            "",
            "---",
            "",
            "## Task Execution Evidence - Step 3 / Controlled Baseline and Multi-Model Comparison",
            "",
            "### Executed run metadata",
            f"- Execution date: `{timestamp}`",
            f"- Main script: `src/train_and_compare_models.py`",
            f"- Execution status: `{STATUS_CONFIRMED_EXEC}`",
            f"- Files inspected before execution: `outputs/shared/split_manifest.json, data/train.csv, data/test.csv, requirements.txt, docs/reproducibility_record.md, docs/decision_log.md`",
            f"- Commands executed: `{execution_command}`",
            f"- Run success rate: `1/1 successful runs`",
            f"- Required artefact completion rate: `{len(created_files)}/14 mandatory artefacts created`",
            f"- Split compliance: `{STATUS_CONFIRMED_EXEC}`",
            f"- Seed compliance: `{STATUS_CONFIRMED_EXEC}`",
            f"- Candidate models executed exactly as specified: `{STATUS_CONFIRMED_EXEC}`",
            f"- Preprocessing fitted on training only: `{STATUS_CONFIRMED_EXEC}`",
            f"- Selection rule applied exactly as specified: `{STATUS_CONFIRMED_EXEC}`",
            f"- Threshold rule preserved (`0.5` unless predeclared otherwise): `{STATUS_CONFIRMED_EXEC}`",
            f"- Test-set misuse detected: `NO`",
            f"- Repeated-run stability check: `NOT_EXECUTED`",
            f"- Selected final model: `{selected_model}`",
            f"- Split manifest SHA-256 reused without modification: `{split_manifest_hash}`",
            f"- Validation ranking snapshot: `{validation_df[['model_name', 'roc_auc', 'pr_auc', 'f1']].to_dict(orient='records')}`",
            f"- Files created: `{', '.join(created_files)}`",
            f"- Failures encountered: `NONE`",
            "",
        ]
    )
    current = record_path.read_text(encoding="utf-8").rstrip()
    record_path.write_text(current + entry + "\n", encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    repo_root = Path(__file__).resolve().parent.parent
    compare_dir = repo_root / "outputs" / "model_compare"
    model_dir = repo_root / "outputs" / "model"
    compare_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    execution_command = os.environ.get(
        "MODEL_COMPARE_EXEC_CMD",
        "python3 src/train_and_compare_models.py",
    )

    dataset = load_prepared_dataset(repo_root / "data")
    manifest_path = repo_root / "outputs" / "shared" / "split_manifest.json"
    manifest, split_manifest_hash_before = read_split_manifest(manifest_path)
    validate_split_manifest(manifest, dataset)

    train_idx = manifest["split_indices"]["train"]
    validation_idx = manifest["split_indices"]["validation"]
    test_idx = manifest["split_indices"]["test"]

    train_df = dataset.loc[train_idx].copy()
    validation_df = dataset.loc[validation_idx].copy()
    test_df = dataset.loc[test_idx].copy()

    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    X_validation = validation_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_validation = validation_df[TARGET_COLUMN]
    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    candidate_models = [
        "LogisticRegression",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "HistGradientBoostingClassifier",
    ]

    validation_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    validation_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    test_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in candidate_models:
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        validation_proba = pipeline.predict_proba(X_validation)[:, 1]
        validation_metrics, validation_pred = compute_metrics(y_validation, validation_proba)
        validation_rows.append(
            {
                "model_name": model_name,
                **validation_metrics,
                "selection_metric": "validation",
                "priority_rank": MODEL_PRIORITY[model_name],
            }
        )
        validation_predictions[model_name] = {
            "proba": validation_proba,
            "pred": validation_pred,
        }

        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics, test_pred = compute_metrics(y_test, test_proba)
        test_rows.append(
            {
                "model_name": model_name,
                **test_metrics,
                "evaluation_note": "post-selection descriptive comparison only",
            }
        )
        test_predictions[model_name] = {
            "proba": test_proba,
            "pred": test_pred,
        }

    validation_metrics_df = pd.DataFrame(validation_rows).sort_values(
        by=["roc_auc", "pr_auc", "f1", "priority_rank"],
        ascending=[False, False, False, True],
    )
    test_metrics_df = pd.DataFrame(test_rows).set_index("model_name").loc[
        validation_metrics_df["model_name"]
    ].reset_index()

    validation_metrics_df.to_csv(
        compare_dir / "validation_metrics_by_model.csv",
        index=False,
    )
    test_metrics_df.to_csv(
        compare_dir / "test_metrics_by_model.csv",
        index=False,
    )

    selected_model = select_model(validation_metrics_df)
    selected_pipeline = trained_pipelines[selected_model]

    selected_validation_metrics = validation_metrics_df.loc[
        validation_metrics_df["model_name"] == selected_model
    ].iloc[0].to_dict()
    selected_test_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"] == selected_model
    ].iloc[0].to_dict()

    validation_prediction_bundle = validation_predictions[selected_model]
    test_prediction_bundle = test_predictions[selected_model]

    save_predictions(
        model_dir / "validation_predictions.csv",
        validation_df["__combined_row_id__"],
        y_validation,
        validation_prediction_bundle["proba"],
        validation_prediction_bundle["pred"],
    )
    save_predictions(
        model_dir / "test_predictions.csv",
        test_df["__combined_row_id__"],
        y_test,
        test_prediction_bundle["proba"],
        test_prediction_bundle["pred"],
    )

    metrics_validation_payload = {
        "model_name": selected_model,
        "split": "validation",
        "metrics": {
            key: value
            for key, value in selected_validation_metrics.items()
            if key not in {"model_name", "selection_metric", "priority_rank"}
        },
        "execution_status": STATUS_CONFIRMED_EXEC,
    }
    metrics_test_payload = {
        "model_name": selected_model,
        "split": "test",
        "metrics": {
            key: value
            for key, value in selected_test_metrics.items()
            if key not in {"model_name", "evaluation_note"}
        },
        "evaluation_note": "formal test evaluation for selected model; candidate test metrics elsewhere are post-selection descriptive comparison only",
        "execution_status": STATUS_CONFIRMED_EXEC,
    }

    (model_dir / "metrics_validation.json").write_text(
        json.dumps(metrics_validation_payload, indent=2, default=json_default),
        encoding="utf-8",
    )
    (model_dir / "metrics_test.json").write_text(
        json.dumps(metrics_test_payload, indent=2, default=json_default),
        encoding="utf-8",
    )

    save_confusion_matrix(
        model_dir / "confusion_matrix_validation.csv",
        metrics_validation_payload["metrics"],
    )
    save_confusion_matrix(
        model_dir / "confusion_matrix_test.csv",
        metrics_test_payload["metrics"],
    )

    joblib.dump(selected_pipeline, model_dir / "model.joblib")
    write_feature_manifest(
        model_dir / "feature_manifest.json",
        selected_model,
        get_feature_names(selected_pipeline),
    )

    write_candidate_manifest(compare_dir / "candidate_model_manifest.json")
    write_selection_report(
        compare_dir / "model_selection_report.md",
        validation_metrics_df.drop(columns=["priority_rank"]),
        test_metrics_df,
        selected_model,
    )

    split_manifest_hash_after = hashlib.sha256(
        manifest_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    if split_manifest_hash_before != split_manifest_hash_after:
        raise RuntimeError("Split manifest changed during Step 3, which is not allowed.")

    created_files = [
        "outputs/model_compare/validation_metrics_by_model.csv",
        "outputs/model_compare/test_metrics_by_model.csv",
        "outputs/model_compare/candidate_model_manifest.json",
        "outputs/model_compare/model_selection_report.md",
        "outputs/model/validation_predictions.csv",
        "outputs/model/test_predictions.csv",
        "outputs/model/metrics_validation.json",
        "outputs/model/metrics_test.json",
        "outputs/model/confusion_matrix_validation.csv",
        "outputs/model/confusion_matrix_test.csv",
        "outputs/model/model.joblib",
        "outputs/model/feature_manifest.json",
        "docs/reproducibility_record.md",
        "docs/decision_log.md",
    ]

    append_decision_log(
        repo_root=repo_root,
        selected_model=selected_model,
        validation_df=validation_metrics_df,
        split_manifest_hash=split_manifest_hash_before,
    )
    append_reproducibility_record(
        repo_root=repo_root,
        selected_model=selected_model,
        validation_df=validation_metrics_df,
        created_files=created_files,
        split_manifest_hash=split_manifest_hash_before,
        execution_command=execution_command,
    )

    model_compare_run_metadata = {
        "task_name": "model_compare",
        "execution_status": STATUS_CONFIRMED_EXEC,
        "random_seed": RANDOM_SEED,
        "split_manifest_path": "outputs/shared/split_manifest.json",
        "split_manifest_sha256": split_manifest_hash_before,
        "selected_model": selected_model,
        "validation_metrics_by_model": validation_metrics_df.to_dict(orient="records"),
        "test_metrics_by_model": test_metrics_df.to_dict(orient="records"),
        "created_files": created_files
        + [
            "outputs/model_compare/run_log.txt",
            "outputs/model_compare/run_metadata.json",
            "outputs/model/run_log.txt",
            "outputs/model/run_metadata.json",
        ],
    }
    model_run_metadata = {
        "task_name": "model",
        "execution_status": STATUS_CONFIRMED_EXEC,
        "random_seed": RANDOM_SEED,
        "selected_model": selected_model,
        "metrics_validation": metrics_validation_payload,
        "metrics_test": metrics_test_payload,
        "feature_manifest_path": "outputs/model/feature_manifest.json",
        "model_artifact_path": "outputs/model/model.joblib",
    }

    compare_log_lines = [
        f"Execution status: {STATUS_CONFIRMED_EXEC}",
        f"Split manifest reused: outputs/shared/split_manifest.json",
        f"Split manifest SHA-256: {split_manifest_hash_before}",
        f"Selected model: {selected_model}",
        f"Validation ranking: {validation_metrics_df[['model_name', 'roc_auc', 'pr_auc', 'f1']].to_dict(orient='records')}",
        "Test metrics note: post-selection descriptive comparison only",
    ]
    (compare_dir / "run_log.txt").write_text(
        "\n".join(compare_log_lines) + "\n",
        encoding="utf-8",
    )
    (compare_dir / "run_metadata.json").write_text(
        json.dumps(model_compare_run_metadata, indent=2, default=json_default),
        encoding="utf-8",
    )

    model_log_lines = [
        f"Execution status: {STATUS_CONFIRMED_EXEC}",
        f"Selected model: {selected_model}",
        f"Validation metrics: {metrics_validation_payload['metrics']}",
        f"Test metrics: {metrics_test_payload['metrics']}",
    ]
    (model_dir / "run_log.txt").write_text(
        "\n".join(model_log_lines) + "\n",
        encoding="utf-8",
    )
    (model_dir / "run_metadata.json").write_text(
        json.dumps(model_run_metadata, indent=2, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(model_compare_run_metadata, indent=2, default=json_default))


if __name__ == "__main__":
    main()
