from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.broken_pipeline import run_pipeline

OUTPUT_DIR = ROOT / "outputs" / "debug_model"


def stable_hash(frame: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(frame, index=False).sum())


def main() -> None:
    runs: list[dict[str, object]] = []
    first_results = None

    for run_index in range(1, 4):
        current_results = run_pipeline(write_outputs=(run_index == 1))
        if first_results is None:
            first_results = current_results
        runs.append(
            {
                "run_index": run_index,
                "status": "success",
                "validation_metrics": current_results["validation_metrics"],
                "test_metrics": current_results["test_metrics"],
                "validation_predictions_hash": stable_hash(current_results["validation_predictions"]),
                "test_predictions_hash": stable_hash(current_results["test_predictions"]),
                "validation_confusion_hash": stable_hash(current_results["validation_confusion"].reset_index()),
                "test_confusion_hash": stable_hash(current_results["test_confusion"].reset_index()),
            }
        )

    validation_serialized = [json.dumps(run["validation_metrics"], sort_keys=True) for run in runs]
    test_serialized = [json.dumps(run["test_metrics"], sort_keys=True) for run in runs]
    validation_frame = pd.DataFrame([run["validation_metrics"] for run in runs])
    test_frame = pd.DataFrame([run["test_metrics"] for run in runs])

    summary = {
        "status": "CONFIRMED BY EXECUTION",
        "attempted_runs": 3,
        "successful_runs": 3,
        "metrics_stable_across_repeated_runs": len(set(validation_serialized)) == 1 and len(set(test_serialized)) == 1,
        "prediction_outputs_identical_across_repeated_runs": len({run["validation_predictions_hash"] for run in runs}) == 1
        and len({run["test_predictions_hash"] for run in runs}) == 1,
        "confusion_matrices_identical_across_repeated_runs": len({run["validation_confusion_hash"] for run in runs}) == 1
        and len({run["test_confusion_hash"] for run in runs}) == 1,
        "max_validation_roc_auc_difference": float(validation_frame["roc_auc"].max() - validation_frame["roc_auc"].min()),
        "max_validation_f1_difference": float(validation_frame["f1_score"].max() - validation_frame["f1_score"].min()),
        "max_test_roc_auc_difference": float(test_frame["roc_auc"].max() - test_frame["roc_auc"].min()),
        "max_test_f1_difference": float(test_frame["f1_score"].max() - test_frame["f1_score"].min()),
        "runs": runs,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "repeat_run_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
