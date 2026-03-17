# Benchmark Summary

## Artefact Coverage

| Workflow area | Artefacts present | Status |
| --- | --- | --- |
| EDA | `outputs/shared/split_manifest.json`, `outputs/eda/`, `docs/eda_summary.md` | confirmed by execution |
| Model comparison | `outputs/model_compare/validation_metrics_by_model.csv`, `outputs/model_compare/test_metrics_by_model.csv`, `outputs/model_compare/candidate_model_manifest.json`, `outputs/model_compare/model_selection_report.md` | confirmed by execution |
| Selected-model evaluation | `outputs/model/metrics_validation.json`, `outputs/model/metrics_test.json`, prediction files, confusion matrices, feature manifest, `model.joblib` | confirmed by execution |
| Debugging review | `src/broken_pipeline_fixed.py`, `outputs/debug/` artefacts, debug metrics and metadata | confirmed by execution |
| Documentation | `README.md`, `docs/model_card.md`, `docs/benchmark_summary.md`, `docs/reproducibility_record.md`, `docs/decision_log.md` | confirmed by file inspection |

## Claim Status

### Confirmed by file inspection

- `requirements.txt` defines the pinned dependency boundary and recommends Python `3.11.x`.
- The raw data are available at the repository root, with `data/train.csv` and `data/test.csv` exposed as symlinks.
- The active audit files are under `docs/`.
- The repository also contains a root-level `reproducibility_record.md`, which is separate from `docs/reproducibility_record.md`.

### Confirmed by execution

- Step 2 created the canonical split manifest with seed `42` and wrote EDA artefacts.
- Step 3 reused the same split manifest and executed all four candidate models.
- The selected final model is `HistGradientBoostingClassifier`.
- Candidate-model test metrics are labelled as post-selection descriptive comparison only.
- Step 4 reused the canonical split while correcting the broken Logistic Regression baseline.

### Unverified

- standalone checked-in command dedicated only to final-model evaluation
- repeated-run stability
- calibration and fairness metrics
- automated test coverage
- deployment monitoring thresholds
- runtime
- token usage
- monetary cost

## Missing Evidence for Human Review

- Confirm whether delay features are acceptable for the intended prediction-time setting.
- Decide whether the duplicate root-level `reproducibility_record.md` should be retired or reconciled with `docs/reproducibility_record.md`.
- Verify whether subgroup performance, fairness review, or external validation are required before any non-benchmark use.
- Confirm whether a separate final-evaluation entrypoint is needed for future workflow clarity.

## Comparison Placeholders

| Item | Value |
| --- | --- |
| Runtime | `UNKNOWN` |
| Token usage | `UNKNOWN` |
| Monetary cost | `UNKNOWN` |
