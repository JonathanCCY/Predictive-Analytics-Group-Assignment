"""
Corrected version of broken_pipeline.py
Original: project root broken_pipeline.py (pre-provided, deliberately broken)
Fixes applied: see docs/decision_log.md (D-014 through D-028)

Preserves original intent: LogisticRegression baseline with median impute + StandardScaler
for numeric features and most_frequent impute + OneHotEncoder for categorical features.
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DEBUG = os.path.join(ROOT, "outputs", "debug")
OUT_SHARED = os.path.join(ROOT, "outputs", "shared")

os.makedirs(OUT_DEBUG, exist_ok=True)

SEED = 42
np.random.seed(SEED)
THRESHOLD = 0.5

run_log_lines = []


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    run_log_lines.append(line)


def evaluate_split(pipeline, X_data, y_data, threshold=THRESHOLD):
    """Compute all required metrics for a fitted pipeline on a given split."""
    y_proba = pipeline.predict_proba(X_data)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "ROC-AUC": round(roc_auc_score(y_data, y_proba), 6),
        "PR-AUC": round(average_precision_score(y_data, y_proba), 6),
        "Accuracy": round(accuracy_score(y_data, y_pred), 6),
        "Precision": round(precision_score(y_data, y_pred, zero_division=0), 6),
        "Recall": round(recall_score(y_data, y_pred, zero_division=0), 6),
        "F1": round(f1_score(y_data, y_pred, zero_division=0), 6),
    }
    cm = confusion_matrix(y_data, y_pred)
    return metrics, cm, y_pred, y_proba


def run_pipeline():
    # ── 1. Load split manifest (reuse canonical split) ─────────────────
    manifest_path = os.path.join(OUT_SHARED, "split_manifest.json")
    if not os.path.exists(manifest_path):
        log(f"FATAL: Split manifest not found at {manifest_path}. Aborting.")
        sys.exit(1)

    with open(manifest_path) as f:
        split_manifest = json.load(f)
    log(f"Loaded split manifest from {manifest_path}")

    # ── 2. Load and combine data ───────────────────────────────────────
    log("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Combine datasets
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    log(f"  Combined shape: {df.shape}")

    # ── FIX B1: Drop unnamed index column and id ───────────────────────
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
    df.drop(columns=unnamed_cols + ["id"], inplace=True)
    log(f"  Dropped columns: {unnamed_cols + ['id']}")

    # ── 3. Encode target ───────────────────────────────────────────────
    log("Preprocessing data...")
    df["satisfaction"] = df["satisfaction"].map(
        {"satisfied": 1, "neutral or dissatisfied": 0}
    )
    assert df["satisfaction"].isna().sum() == 0, "Unmapped target values!"

    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]

    # ── 4. Reconstruct splits from manifest (FIX B3, B4, B5) ──────────
    log("Reconstructing splits from manifest indices...")
    train_idx = split_manifest["splits"]["train"]["indices"]
    val_idx = split_manifest["splits"]["validation"]["indices"]
    test_idx = split_manifest["splits"]["test"]["indices"]

    # Verify integrity
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    log(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ── 5. Define feature lists ────────────────────────────────────────
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    log(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
    log(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

    # ── 6. Build preprocessing pipeline (same structure as original) ───
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ── FIX B6, B7, B8: Correct LogisticRegression parameters ─────────
    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")

    full_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", model)]
    )

    # ── 7. FIX B2/B5: Fit on training data ONLY ───────────────────────
    log("Training model...")
    # FIX B9: fit on X_train, y_train (not y_temp)
    full_pipeline.fit(X_train, y_train)
    log("  Model fitted on training data only.")

    # ── 8. Evaluate on validation ──────────────────────────────────────
    log("Evaluating on validation set...")
    val_metrics, val_cm, val_pred, val_proba = evaluate_split(
        full_pipeline, X_val, y_val
    )
    log(f"  Validation Accuracy: {val_metrics['Accuracy']:.4f}")
    log(f"  Validation ROC-AUC: {val_metrics['ROC-AUC']:.6f}")
    log(f"  Validation F1: {val_metrics['F1']:.6f}")
    log("  Validation Classification Report:")
    for line in classification_report(y_val, val_pred).split("\n"):
        log(f"    {line}")

    # ── 9. FIX B10/B11: Evaluate on test set ──────────────────────────
    log("Evaluating on test set...")
    test_metrics, test_cm, test_pred, test_proba = evaluate_split(
        full_pipeline, X_test, y_test
    )
    log(f"  Test Accuracy: {test_metrics['Accuracy']:.4f}")
    log(f"  Test ROC-AUC: {test_metrics['ROC-AUC']:.6f}")
    log(f"  Test F1: {test_metrics['F1']:.6f}")
    log("  Test Classification Report:")
    for line in classification_report(y_test, test_pred).split("\n"):
        log(f"    {line}")

    # ── 10. Save outputs (FIX B12, B13, B14, B15) ─────────────────────
    log("Saving outputs to outputs/debug/...")

    # FIX B12/B13: Use joblib in binary mode, not pickle in text mode
    joblib.dump(full_pipeline, os.path.join(OUT_DEBUG, "model.joblib"))

    # FIX B15: Save metrics as JSON
    with open(os.path.join(OUT_DEBUG, "metrics_validation.json"), "w") as f:
        json.dump(
            {
                "model": "LogisticRegression",
                "split": "validation",
                "threshold": THRESHOLD,
                "metrics": val_metrics,
                "execution_status": "CONFIRMED_BY_EXECUTION",
            },
            f,
            indent=2,
        )

    with open(os.path.join(OUT_DEBUG, "metrics_test.json"), "w") as f:
        json.dump(
            {
                "model": "LogisticRegression",
                "split": "test",
                "threshold": THRESHOLD,
                "metrics": test_metrics,
                "execution_status": "CONFIRMED_BY_EXECUTION",
            },
            f,
            indent=2,
        )

    # Predictions
    pd.DataFrame(
        {
            "index": val_idx,
            "y_true": y_val.values,
            "y_pred": val_pred,
            "y_proba": val_proba,
        }
    ).to_csv(os.path.join(OUT_DEBUG, "validation_predictions.csv"), index=False)

    pd.DataFrame(
        {
            "index": test_idx,
            "y_true": y_test.values,
            "y_pred": test_pred,
            "y_proba": test_proba,
        }
    ).to_csv(os.path.join(OUT_DEBUG, "test_predictions.csv"), index=False)

    # Confusion matrices
    pd.DataFrame(
        val_cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(os.path.join(OUT_DEBUG, "confusion_matrix_validation.csv"))

    pd.DataFrame(
        test_cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(os.path.join(OUT_DEBUG, "confusion_matrix_test.csv"))

    # Split manifest reference (copy for audit trail)
    with open(os.path.join(OUT_DEBUG, "split_manifest.json"), "w") as f:
        json.dump(
            {
                "note": "Reference copy — canonical source is outputs/shared/split_manifest.json",
                "canonical_path": "outputs/shared/split_manifest.json",
                "random_seed": split_manifest["random_seed"],
                "train_rows": split_manifest["splits"]["train"]["n_rows"],
                "val_rows": split_manifest["splits"]["validation"]["n_rows"],
                "test_rows": split_manifest["splits"]["test"]["n_rows"],
            },
            f,
            indent=2,
        )

    # Feature manifest
    try:
        transformed_names = full_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out().tolist()
    except Exception:
        transformed_names = "UNKNOWN"

    with open(os.path.join(OUT_DEBUG, "feature_manifest.json"), "w") as f:
        json.dump(
            {
                "model": "LogisticRegression",
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "total_input_features": len(numeric_features)
                + len(categorical_features),
                "transformed_feature_names": transformed_names,
                "total_transformed_features": len(transformed_names)
                if isinstance(transformed_names, list)
                else "UNKNOWN",
            },
            f,
            indent=2,
        )

    # Run metadata
    with open(os.path.join(OUT_DEBUG, "run_metadata.json"), "w") as f:
        json.dump(
            {
                "task": "Step 4 - Debug Broken Pipeline",
                "original_file": "broken_pipeline.py",
                "corrected_file": "src/broken_pipeline_fixed.py",
                "execution_date": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "python_version": sys.version,
                "random_seed": SEED,
                "threshold": THRESHOLD,
                "model": "LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')",
                "split_manifest_reused": True,
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "test_rows": len(y_test),
                "execution_status": "CONFIRMED_BY_EXECUTION",
            },
            f,
            indent=2,
        )

    # Run log
    with open(os.path.join(OUT_DEBUG, "run_log.txt"), "w") as f:
        f.write("\n".join(run_log_lines))

    log("Pipeline finished successfully!")


if __name__ == "__main__":
    run_pipeline()
