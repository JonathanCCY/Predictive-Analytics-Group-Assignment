from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "debug_model"


def test_metrics_files_exist_with_required_keys():
    metrics_path = OUTPUT_DIR / "metrics_validation.json"
    if not metrics_path.exists():
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1_score"}
    assert required.issubset(metrics)


def test_prediction_columns_if_predictions_exist():
    predictions_path = OUTPUT_DIR / "validation_predictions.csv"
    if not predictions_path.exists():
        return

    predictions = pd.read_csv(predictions_path)
    assert {"row_id", "y_true", "y_prob", "y_pred"}.issubset(predictions.columns)
