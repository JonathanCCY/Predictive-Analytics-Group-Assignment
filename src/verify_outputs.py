from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "debug_model"
EXPECTED_FILES = [
    OUTPUT_DIR / "model.joblib",
    OUTPUT_DIR / "metrics_validation.json",
    OUTPUT_DIR / "metrics_test.json",
    OUTPUT_DIR / "validation_predictions.csv",
    OUTPUT_DIR / "test_predictions.csv",
    OUTPUT_DIR / "feature_manifest.json",
    OUTPUT_DIR / "split_manifest.json",
    OUTPUT_DIR / "run_metadata.json",
    OUTPUT_DIR / "confusion_matrix_validation.csv",
    OUTPUT_DIR / "confusion_matrix_test.csv",
    OUTPUT_DIR / "run_log.txt",
]


def verify_outputs() -> dict[str, object]:
    file_checks: dict[str, dict[str, object]] = {}
    all_present = True
    for path in EXPECTED_FILES:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        file_checks[str(path.relative_to(ROOT))] = {
            "exists": exists,
            "non_empty": size > 0,
            "size_bytes": size,
        }
        all_present = all_present and exists and size > 0

    content_checks: dict[str, object] = {}
    metrics_path = OUTPUT_DIR / "metrics_validation.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        required_metric_keys = {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1_score"}
        content_checks["validation_metric_keys_present"] = required_metric_keys.issubset(metrics)

    predictions_path = OUTPUT_DIR / "validation_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        content_checks["validation_prediction_columns"] = predictions.columns.tolist()

    return {
        "status": "CONFIRMED BY EXECUTION" if all_present else "FAILED DURING EXECUTION",
        "expected_files": [str(path.relative_to(ROOT)) for path in EXPECTED_FILES],
        "file_checks": file_checks,
        "content_checks": content_checks,
    }


if __name__ == "__main__":
    print(json.dumps(verify_outputs(), indent=2))
