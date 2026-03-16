"""
Step 3: Controlled Baseline Model Training and Comparison Harness
- Reuses canonical split from outputs/shared/split_manifest.json
- Trains 4 fixed candidate models with specified preprocessing
- Selects final model by validation ROC-AUC (tie-breakers: PR-AUC, F1, priority order)
- Reports test metrics as post-selection descriptive comparison
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_COMPARE = os.path.join(ROOT, "outputs", "model_compare")
OUT_MODEL = os.path.join(ROOT, "outputs", "model")
OUT_SHARED = os.path.join(ROOT, "outputs", "shared")

os.makedirs(OUT_COMPARE, exist_ok=True)
os.makedirs(OUT_MODEL, exist_ok=True)

SEED = 42
np.random.seed(SEED)
THRESHOLD = 0.5

run_log_lines = []


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    run_log_lines.append(line)


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD SPLIT MANIFEST
# ══════════════════════════════════════════════════════════════════════════
manifest_path = os.path.join(OUT_SHARED, "split_manifest.json")
if not os.path.exists(manifest_path):
    log(f"FATAL: Split manifest not found at {manifest_path}. Aborting.")
    sys.exit(1)

with open(manifest_path) as f:
    split_manifest = json.load(f)

log(f"Loaded split manifest from {manifest_path}")
log(f"  Seed: {split_manifest['random_seed']}")
log(
    f"  Train: {split_manifest['splits']['train']['n_rows']}, "
    f"Val: {split_manifest['splits']['validation']['n_rows']}, "
    f"Test: {split_manifest['splits']['test']['n_rows']}"
)

# ══════════════════════════════════════════════════════════════════════════
# 2. RELOAD AND PREPARE DATA (same steps as Step 2)
# ══════════════════════════════════════════════════════════════════════════
log("Loading and preparing data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Drop unnamed first column and id
unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
df.drop(columns=unnamed_cols + ["id"], inplace=True)

# Encode target
df["satisfaction"] = df["satisfaction"].map(
    {"satisfied": 1, "neutral or dissatisfied": 0}
)
assert df["satisfaction"].isna().sum() == 0, "Unmapped target values!"

log(f"  Combined shape after drops: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 3. RECONSTRUCT SPLITS FROM MANIFEST INDICES
# ══════════════════════════════════════════════════════════════════════════
log("Reconstructing splits from manifest indices...")
train_idx = split_manifest["splits"]["train"]["indices"]
val_idx = split_manifest["splits"]["validation"]["indices"]
test_idx = split_manifest["splits"]["test"]["indices"]

# Verify no overlap
assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), "Index count mismatch!"

X = df.drop(columns=["satisfaction"])
y = df["satisfaction"]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val, y_val = X.loc[val_idx], y.loc[val_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

log(
    f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}"
)

# Verify against manifest counts
assert len(y_train) == split_manifest["splits"]["train"]["n_rows"]
assert len(y_val) == split_manifest["splits"]["validation"]["n_rows"]
assert len(y_test) == split_manifest["splits"]["test"]["n_rows"]
log("  Split index reconstruction verified against manifest.")

# ══════════════════════════════════════════════════════════════════════════
# 4. DEFINE FEATURE LISTS
# ══════════════════════════════════════════════════════════════════════════
numeric_features = [
    "Age", "Flight Distance",
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
]

categorical_features = ["Gender", "Customer Type", "Type of Travel", "Class"]

# Verify all features present
for col in numeric_features + categorical_features:
    assert col in X_train.columns, f"Missing feature: {col}"
log(f"  Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")


# ══════════════════════════════════════════════════════════════════════════
# 5. DEFINE MODEL-SPECIFIC PIPELINES
# ══════════════════════════════════════════════════════════════════════════
def build_pipeline(model_name):
    """Build the full preprocessing + model pipeline for a given model name."""

    if model_name == "LogisticRegression":
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")

    elif model_name == "RandomForestClassifier":
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_name == "ExtraTreesClassifier":
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    elif model_name == "HistGradientBoostingClassifier":
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ])
        model = HistGradientBoostingClassifier(max_iter=100, random_state=42)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    return pipeline


# ══════════════════════════════════════════════════════════════════════════
# 6. EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════
def evaluate(pipeline, X_data, y_data, threshold=THRESHOLD):
    """Compute all required metrics for a fitted pipeline."""
    y_proba = pipeline.predict_proba(X_data)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc = roc_auc_score(y_data, y_proba)
    pr = average_precision_score(y_data, y_proba)
    acc = accuracy_score(y_data, y_pred)
    prec = precision_score(y_data, y_pred, zero_division=0)
    rec = recall_score(y_data, y_pred, zero_division=0)
    f1 = f1_score(y_data, y_pred, zero_division=0)
    cm = confusion_matrix(y_data, y_pred)

    metrics = {
        "ROC-AUC": round(roc, 6),
        "PR-AUC": round(pr, 6),
        "Accuracy": round(acc, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1": round(f1, 6),
    }
    return metrics, cm, y_pred, y_proba


# ══════════════════════════════════════════════════════════════════════════
# 7. TRAIN AND EVALUATE ALL CANDIDATES
# ══════════════════════════════════════════════════════════════════════════
MODEL_NAMES = [
    "LogisticRegression",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "HistGradientBoostingClassifier",
]

# Priority order for tie-breaking (lower index = higher priority)
PRIORITY_ORDER = {
    "LogisticRegression": 0,
    "HistGradientBoostingClassifier": 1,
    "RandomForestClassifier": 2,
    "ExtraTreesClassifier": 3,
}

val_results = {}
test_results = {}
fitted_pipelines = {}

for name in MODEL_NAMES:
    log(f"Training {name}...")
    pipeline = build_pipeline(name)

    # Fit on training data only
    pipeline.fit(X_train, y_train)
    fitted_pipelines[name] = pipeline

    # Evaluate on validation
    val_metrics, val_cm, val_pred, val_proba = evaluate(pipeline, X_val, y_val)
    val_results[name] = {
        "metrics": val_metrics,
        "confusion_matrix": val_cm,
        "predictions": val_pred,
        "probabilities": val_proba,
    }
    log(f"  Validation ROC-AUC: {val_metrics['ROC-AUC']:.6f}, "
        f"PR-AUC: {val_metrics['PR-AUC']:.6f}, F1: {val_metrics['F1']:.6f}")

    # Evaluate on test (post-selection descriptive only)
    test_metrics, test_cm, test_pred, test_proba = evaluate(pipeline, X_test, y_test)
    test_results[name] = {
        "metrics": test_metrics,
        "confusion_matrix": test_cm,
        "predictions": test_pred,
        "probabilities": test_proba,
    }
    log(f"  Test ROC-AUC: {test_metrics['ROC-AUC']:.6f} (post-selection descriptive)")

log("All 4 models trained and evaluated.")

# ══════════════════════════════════════════════════════════════════════════
# 8. MODEL SELECTION (validation-governed)
# ══════════════════════════════════════════════════════════════════════════
log("Applying model selection rule...")


def selection_key(name):
    """
    Returns a tuple for sorting: higher is better for ROC-AUC, PR-AUC, F1;
    lower priority index is better. We negate the metrics for ascending sort.
    """
    m = val_results[name]["metrics"]
    return (
        -m["ROC-AUC"],
        -m["PR-AUC"],
        -m["F1"],
        PRIORITY_ORDER[name],
    )


ranked = sorted(MODEL_NAMES, key=selection_key)
selected_model_name = ranked[0]

log(f"  Selection ranking (best first):")
for i, name in enumerate(ranked):
    m = val_results[name]["metrics"]
    marker = " <-- SELECTED" if i == 0 else ""
    log(f"    {i+1}. {name}: ROC-AUC={m['ROC-AUC']:.6f}, "
        f"PR-AUC={m['PR-AUC']:.6f}, F1={m['F1']:.6f}{marker}")

log(f"  Final selected model: {selected_model_name}")

# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE COMPARISON LAYER ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════
log("Saving comparison layer artefacts...")

# validation_metrics_by_model.csv
val_metrics_rows = []
for name in MODEL_NAMES:
    row = {"model": name}
    row.update(val_results[name]["metrics"])
    val_metrics_rows.append(row)
val_metrics_df = pd.DataFrame(val_metrics_rows)
val_metrics_df.to_csv(
    os.path.join(OUT_COMPARE, "validation_metrics_by_model.csv"), index=False
)

# test_metrics_by_model.csv (post-selection descriptive)
test_metrics_rows = []
for name in MODEL_NAMES:
    row = {"model": name, "label": "post-selection descriptive comparison"}
    row.update(test_results[name]["metrics"])
    test_metrics_rows.append(row)
test_metrics_df = pd.DataFrame(test_metrics_rows)
test_metrics_df.to_csv(
    os.path.join(OUT_COMPARE, "test_metrics_by_model.csv"), index=False
)

# candidate_model_manifest.json
candidate_manifest = {
    "candidate_models": MODEL_NAMES,
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
    "random_seed": SEED,
    "selection_metric": "validation ROC-AUC",
    "tie_break_rules": ["PR-AUC", "F1 at 0.5", "fixed priority order"],
    "threshold_rule": "fixed at 0.5",
    "dependency_scope": "requirements.txt",
    "selected_model": selected_model_name,
    "selection_ranking": [
        {
            "rank": i + 1,
            "model": name,
            "val_ROC_AUC": val_results[name]["metrics"]["ROC-AUC"],
            "val_PR_AUC": val_results[name]["metrics"]["PR-AUC"],
            "val_F1": val_results[name]["metrics"]["F1"],
        }
        for i, name in enumerate(ranked)
    ],
    "execution_status": "CONFIRMED_BY_EXECUTION",
}
with open(os.path.join(OUT_COMPARE, "candidate_model_manifest.json"), "w") as f:
    json.dump(candidate_manifest, f, indent=2)

# model_selection_report.md
report_lines = [
    "# Model Selection Report",
    "",
    "## Candidate Models",
    "",
    "| Model | Preprocessing |",
    "|-------|--------------|",
]
for name in MODEL_NAMES:
    report_lines.append(
        f"| {name} | {candidate_manifest['preprocessing_by_model'][name]} |"
    )
report_lines += [
    "",
    "## Validation Metrics",
    "",
    "| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |",
    "|-------|---------|--------|----------|-----------|--------|-----|",
]
for name in MODEL_NAMES:
    m = val_results[name]["metrics"]
    report_lines.append(
        f"| {name} | {m['ROC-AUC']:.6f} | {m['PR-AUC']:.6f} | "
        f"{m['Accuracy']:.6f} | {m['Precision']:.6f} | {m['Recall']:.6f} | {m['F1']:.6f} |"
    )
report_lines += [
    "",
    "## Selection Rule",
    "",
    "- Primary: validation ROC-AUC (higher is better)",
    "- Tie-breaker 1: validation PR-AUC",
    "- Tie-breaker 2: validation F1 at threshold 0.5",
    "- Tie-breaker 3: fixed priority order (LR > HGBC > RF > ET)",
    "",
    "## Selection Ranking",
    "",
]
for i, name in enumerate(ranked):
    m = val_results[name]["metrics"]
    marker = " **SELECTED**" if i == 0 else ""
    report_lines.append(
        f"{i+1}. {name} — ROC-AUC={m['ROC-AUC']:.6f}, "
        f"PR-AUC={m['PR-AUC']:.6f}, F1={m['F1']:.6f}{marker}"
    )
report_lines += [
    "",
    f"**Final selected model: {selected_model_name}**",
    "",
    "## Test Metrics (Post-Selection Descriptive Comparison)",
    "",
    "These test metrics are reported **after** model selection and were **not** used for selection.",
    "",
    "| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |",
    "|-------|---------|--------|----------|-----------|--------|-----|",
]
for name in MODEL_NAMES:
    m = test_results[name]["metrics"]
    report_lines.append(
        f"| {name} | {m['ROC-AUC']:.6f} | {m['PR-AUC']:.6f} | "
        f"{m['Accuracy']:.6f} | {m['Precision']:.6f} | {m['Recall']:.6f} | {m['F1']:.6f} |"
    )
report_lines += [
    "",
    "## Threshold",
    "",
    "Classification threshold fixed at 0.5. No threshold tuning was performed.",
]

with open(os.path.join(OUT_COMPARE, "model_selection_report.md"), "w") as f:
    f.write("\n".join(report_lines) + "\n")

log("  Comparison layer artefacts saved.")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE FINAL SELECTED MODEL ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════
log("Saving final model artefacts...")

selected_pipeline = fitted_pipelines[selected_model_name]
sel_val = val_results[selected_model_name]
sel_test = test_results[selected_model_name]

# model.joblib
joblib.dump(selected_pipeline, os.path.join(OUT_MODEL, "model.joblib"))
log("  model.joblib saved.")

# validation_predictions.csv
val_pred_df = pd.DataFrame({
    "index": val_idx,
    "y_true": y_val.values,
    "y_pred": sel_val["predictions"],
    "y_proba": sel_val["probabilities"],
})
val_pred_df.to_csv(os.path.join(OUT_MODEL, "validation_predictions.csv"), index=False)

# test_predictions.csv
test_pred_df = pd.DataFrame({
    "index": test_idx,
    "y_true": y_test.values,
    "y_pred": sel_test["predictions"],
    "y_proba": sel_test["probabilities"],
})
test_pred_df.to_csv(os.path.join(OUT_MODEL, "test_predictions.csv"), index=False)

# metrics_validation.json
with open(os.path.join(OUT_MODEL, "metrics_validation.json"), "w") as f:
    json.dump(
        {
            "model": selected_model_name,
            "split": "validation",
            "threshold": THRESHOLD,
            "metrics": sel_val["metrics"],
            "execution_status": "CONFIRMED_BY_EXECUTION",
        },
        f,
        indent=2,
    )

# metrics_test.json
with open(os.path.join(OUT_MODEL, "metrics_test.json"), "w") as f:
    json.dump(
        {
            "model": selected_model_name,
            "split": "test",
            "label": "post-selection descriptive",
            "threshold": THRESHOLD,
            "metrics": sel_test["metrics"],
            "execution_status": "CONFIRMED_BY_EXECUTION",
        },
        f,
        indent=2,
    )

# confusion_matrix_validation.csv
cm_val = sel_val["confusion_matrix"]
pd.DataFrame(
    cm_val,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(os.path.join(OUT_MODEL, "confusion_matrix_validation.csv"))

# confusion_matrix_test.csv
cm_test = sel_test["confusion_matrix"]
pd.DataFrame(
    cm_test,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(os.path.join(OUT_MODEL, "confusion_matrix_test.csv"))

# feature_manifest.json
preprocessor = selected_pipeline.named_steps["preprocessor"]
# Get transformed feature names
try:
    transformed_names = preprocessor.get_feature_names_out().tolist()
except Exception:
    transformed_names = "UNKNOWN"

feature_manifest = {
    "model": selected_model_name,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "total_input_features": len(numeric_features) + len(categorical_features),
    "transformed_feature_names": transformed_names,
    "total_transformed_features": len(transformed_names) if isinstance(transformed_names, list) else "UNKNOWN",
}
with open(os.path.join(OUT_MODEL, "feature_manifest.json"), "w") as f:
    json.dump(feature_manifest, f, indent=2)

log("  Final model artefacts saved.")

# ══════════════════════════════════════════════════════════════════════════
# 11. SAVE RUN LOGS AND METADATA
# ══════════════════════════════════════════════════════════════════════════

# Run logs for model_compare
with open(os.path.join(OUT_COMPARE, "run_log.txt"), "w") as f:
    f.write("\n".join(run_log_lines))

compare_metadata = {
    "task": "Step 3 - Controlled Baseline Model Training and Comparison",
    "script": "src/train_and_compare_models.py",
    "execution_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "python_version": sys.version,
    "random_seed": SEED,
    "threshold": THRESHOLD,
    "split_manifest_path": "outputs/shared/split_manifest.json",
    "split_manifest_reused": True,
    "train_rows": len(y_train),
    "val_rows": len(y_val),
    "test_rows": len(y_test),
    "candidate_models": MODEL_NAMES,
    "selected_model": selected_model_name,
    "selection_metric": "validation ROC-AUC",
    "execution_status": "CONFIRMED_BY_EXECUTION",
}
with open(os.path.join(OUT_COMPARE, "run_metadata.json"), "w") as f:
    json.dump(compare_metadata, f, indent=2)

# Run logs/metadata for model
with open(os.path.join(OUT_MODEL, "run_log.txt"), "w") as f:
    f.write("\n".join(run_log_lines))

model_metadata = {
    "task": "Step 3 - Final Selected Model",
    "model": selected_model_name,
    "script": "src/train_and_compare_models.py",
    "execution_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "python_version": sys.version,
    "random_seed": SEED,
    "threshold": THRESHOLD,
    "validation_metrics": sel_val["metrics"],
    "test_metrics": sel_test["metrics"],
    "test_label": "post-selection descriptive",
    "execution_status": "CONFIRMED_BY_EXECUTION",
}
with open(os.path.join(OUT_MODEL, "run_metadata.json"), "w") as f:
    json.dump(model_metadata, f, indent=2)

log("Run logs and metadata saved.")
log("=" * 60)
log("Step 3 complete.")
log(f"Selected model: {selected_model_name}")
log(f"Validation ROC-AUC: {sel_val['metrics']['ROC-AUC']:.6f}")
log(f"Test ROC-AUC: {sel_test['metrics']['ROC-AUC']:.6f} (post-selection descriptive)")
