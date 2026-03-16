# Reproducibility Record

## 1. Task
- Task label: `EDA_NOTEBOOK_INLINE`
- Task name: Airline Passenger Satisfaction exploratory data analysis and insight generation
- Date: 2026-03-15
- Main script(s): `Untitled-1.ipynb`

## 2. Execution status
- Status: CONFIRMED BY EXECUTION
- Environment notes: `Untitled-1.ipynb` was executed directly as the self-contained EDA workflow. The notebook generates the required plots, summary tables, data-quality report, and markdown summary from local project files only.
- Manual intervention required: NO

## 3. Quantitative indicators
- Run success rate: 1 / 1
- Required artefact completion rate: 9 / 9
- Split compliance: PASS
- Seed compliance: PASS
- Output stability across repeated runs: metrics stable across repeated runs: NOT EXECUTED; prediction outputs identical across repeated runs: NOT APPLICABLE; file hashes identical across repeated runs: NOT EXECUTED; max ROC-AUC difference across runs: NOT APPLICABLE; max F1 difference across runs: NOT APPLICABLE

## 4. Evidence
- Attempted runs: 1 direct notebook execution of `Untitled-1.ipynb`
- Successful runs: 1
- Required files expected:
  - `outputs/eda/class_balance.png`
  - `outputs/eda/missing_values.png`
  - `outputs/eda/numeric_summary.csv`
  - `outputs/eda/categorical_summary.csv`
  - `outputs/eda/numeric_distributions.png`
  - `outputs/eda/top_target_associations.png`
  - `outputs/eda/correlation_heatmap.png`
  - `outputs/eda/data_quality_report.json`
  - `docs/eda_summary.md`
- Required files created:
  - `outputs/eda/class_balance.png`
  - `outputs/eda/missing_values.png`
  - `outputs/eda/numeric_summary.csv`
  - `outputs/eda/categorical_summary.csv`
  - `outputs/eda/numeric_distributions.png`
  - `outputs/eda/top_target_associations.png`
  - `outputs/eda/correlation_heatmap.png`
  - `outputs/eda/data_quality_report.json`
  - `docs/eda_summary.md`
- Verification performed: YES
- Repeated-run check performed: NO

## 5. Methodological checks
- Test-set misuse: NO
- Leakage risk found: NO
- Label handling correct: YES
- Validation-only threshold/model decision preserved: NOT APPLICABLE
- Split rule preserved: YES
- Random seed documented and applied: YES

## 6. Failures or unresolved issues
- The notebook execution required a non-sandboxed run so Jupyter could start a kernel normally in this environment.

## 7. Assumptions
- `train.csv` and `test.csv` are treated as the original Kaggle source files rather than the benchmark split artefacts.
- The benchmark split was reconstructed by combining the two CSV files and applying a stratified 70/15/15 split with random seed `42`.
- Only the derived training split was used for EDA plots, summaries, and target-aware insight generation.

## 8. Not executed or unverified items
- Repeated direct notebook executions for stability measurement: NOT EXECUTED

## 9. Files created or reviewed
- `Untitled-1.ipynb`
- `docs/eda_summary.md`
- `docs/reproducibility_record.md`
- `outputs/eda/class_balance.png`
- `outputs/eda/missing_values.png`
- `outputs/eda/numeric_summary.csv`
- `outputs/eda/categorical_summary.csv`
- `outputs/eda/numeric_distributions.png`
- `outputs/eda/top_target_associations.png`
- `outputs/eda/correlation_heatmap.png`
- `outputs/eda/data_quality_report.json`
- `outputs/eda/split_manifest.json`

# Reproducibility Record

## 1. Task
- Task label: `DEBUG_BROKEN_PIPELINE`
- Task name: Debug and correct the deliberately broken baseline pipeline
- Date: 2026-03-15
- Main script(s): `broken_pipeline.py`, `src/broken_pipeline.py`, `src/verify_outputs.py`, `src/repeat_run_check.py`, `tests/test_pipeline_outputs.py`

## 2. Execution status
- Status: CONFIRMED BY EXECUTION
- Environment notes: The requested `src/broken_pipeline.py`, `data/train.csv`, `data/test.csv`, and `logs/error_log.txt` paths were not present in this workspace at inspection time. Diagnosis used the available local evidence in `broken_pipeline.py`, `train.csv`, `test.csv`, `agent_reproducibility_protocol_v3.md`, and `requirements.txt`. The corrected pipeline was executed successfully outside the sandbox because an in-sandbox OpenMP shared-memory error prevented a valid end-to-end run there.
- Manual intervention required: NO

## 3. Quantitative indicators
- Run success rate: 4 / 6
- Required artefact completion rate: 11 / 11
- Split compliance: PASS
- Seed compliance: PASS
- Output stability across repeated runs: metrics stable across repeated runs: YES; prediction outputs identical across repeated runs: YES; max ROC-AUC difference across runs: validation=0.0000000000, test=0.0000000000; max F1 difference across runs: validation=0.0000000000, test=0.0000000000

## 4. Evidence
- Attempted runs:
  - `python broken_pipeline.py` before code changes: FAILED DURING EXECUTION with `FileNotFoundError` for `data/train.csv`
  - `python broken_pipeline.py` after code changes inside sandbox: FAILED DURING EXECUTION with OpenMP shared-memory environment error
  - `python broken_pipeline.py` after code changes outside sandbox: CONFIRMED BY EXECUTION
  - `python src/repeat_run_check.py` outside sandbox: 3 / 3 successful repeated runs
- Successful runs: 4
- Required files expected:
  - `outputs/debug_model/model.joblib`
  - `outputs/debug_model/metrics_validation.json`
  - `outputs/debug_model/metrics_test.json`
  - `outputs/debug_model/validation_predictions.csv`
  - `outputs/debug_model/test_predictions.csv`
  - `outputs/debug_model/feature_manifest.json`
  - `outputs/debug_model/split_manifest.json`
  - `outputs/debug_model/run_metadata.json`
  - `outputs/debug_model/confusion_matrix_validation.csv`
  - `outputs/debug_model/confusion_matrix_test.csv`
  - `outputs/debug_model/run_log.txt`
- Required files created:
  - `outputs/debug_model/model.joblib`
  - `outputs/debug_model/metrics_validation.json`
  - `outputs/debug_model/metrics_test.json`
  - `outputs/debug_model/validation_predictions.csv`
  - `outputs/debug_model/test_predictions.csv`
  - `outputs/debug_model/feature_manifest.json`
  - `outputs/debug_model/split_manifest.json`
  - `outputs/debug_model/run_metadata.json`
  - `outputs/debug_model/confusion_matrix_validation.csv`
  - `outputs/debug_model/confusion_matrix_test.csv`
  - `outputs/debug_model/run_log.txt`
- Verification performed: YES
- Repeated-run check performed: YES

## 5. Methodological checks
- Test-set misuse: NO
- Leakage risk found: NO
- Label handling correct: YES
- Validation-only threshold/model decision preserved: YES
- Split rule preserved: YES
- Random seed documented and applied: YES

## 6. Failures or unresolved issues
- Confirmed by execution: the original script failed immediately because it hard-coded non-existent `data/` paths in this workspace.
- Confirmed by file inspection: the original script also contained a target-length mismatch in `model.fit(X_train, y_temp)`, pre-split preprocessing leakage, non-deterministic splitting, invalid “test” reporting that actually printed validation results, and incorrect binary pickle write mode.
- Confirmed by execution: running the corrected script inside the sandbox failed because of an OpenMP shared-memory environment issue rather than a code defect.

## 7. Assumptions
- `train.csv` and `test.csv` at the project root are the intended local source files because the requested `data/` paths were absent.
- The original Kaggle-style source files were combined before applying the benchmark’s custom stratified 70/15/15 split with fixed seed `42`.
- Identifier-style columns (`Unnamed: 0`, `id`) were excluded from modelling as a validity fix that preserves the baseline classification intent while avoiding obvious non-predictive signals.

## 8. Not executed or unverified items
- `logs/error_log.txt`: NOT EXECUTED because the file was not present
- Execution of a pipeline version located specifically at `src/broken_pipeline.py` before repair: UNVERIFIED because that path did not exist initially

## 9. Files created or reviewed
- `broken_pipeline.py`
- `src/broken_pipeline.py`
- `src/verify_outputs.py`
- `src/repeat_run_check.py`
- `tests/test_pipeline_outputs.py`
- `docs/reproducibility_record.md`
- `outputs/debug_model/model.joblib`
- `outputs/debug_model/metrics_validation.json`
- `outputs/debug_model/metrics_test.json`
- `outputs/debug_model/validation_predictions.csv`
- `outputs/debug_model/test_predictions.csv`
- `outputs/debug_model/feature_manifest.json`
- `outputs/debug_model/split_manifest.json`
- `outputs/debug_model/run_metadata.json`
- `outputs/debug_model/confusion_matrix_validation.csv`
- `outputs/debug_model/confusion_matrix_test.csv`
- `outputs/debug_model/run_log.txt`
- `outputs/debug_model/repeat_run_check.json`


# Reproducibility Record

## 1. Task
- Task label: `BASELINE_MODEL_NOTEBOOK`
- Task name: Baseline LogisticRegression modelling and evaluation harness
- Date: 2026-03-15
- Main script(s): `Untitled-1.ipynb`

## 2. Execution status
- Status: CONFIRMED BY EXECUTION
- Environment notes: The baseline modelling pipeline was executed inside `Untitled-1.ipynb` with LogisticRegression, validation and test evaluation, saved artefacts, and a 3-run repeated-run check using the same settings.
- Manual intervention required: NO

## 3. Quantitative indicators
- Run success rate: 3 / 3
- Required artefact completion rate: 11 / 11
- Split compliance: PASS
- Seed compliance: PASS
- Output stability across repeated runs: metrics stable across repeated runs: YES; prediction outputs identical across repeated runs: YES; max ROC-AUC difference across runs: validation=0.0000000000, test=0.0000000000; max F1 difference across runs: validation=0.0000000000, test=0.0000000000

## 4. Evidence
- Attempted runs: 3 baseline pipeline runs inside the notebook under identical settings; the first run saved the official artefacts and all 3 runs were compared for stability.
- Successful runs: 3
- Required files expected:
  - `outputs/model/metrics_validation.json`
  - `outputs/model/metrics_test.json`
  - `outputs/model/validation_predictions.csv`
  - `outputs/model/test_predictions.csv`
  - `outputs/model/model.joblib`
  - `outputs/model/split_manifest.json`
  - `outputs/model/feature_manifest.json`
  - `outputs/model/confusion_matrix_validation.csv`
  - `outputs/model/confusion_matrix_test.csv`
  - `outputs/model/run_metadata.json`
  - `outputs/model/run_log.txt`
- Required files created:
  - `outputs/model/metrics_validation.json`
  - `outputs/model/metrics_test.json`
  - `outputs/model/validation_predictions.csv`
  - `outputs/model/test_predictions.csv`
  - `outputs/model/model.joblib`
  - `outputs/model/split_manifest.json`
  - `outputs/model/feature_manifest.json`
  - `outputs/model/confusion_matrix_validation.csv`
  - `outputs/model/confusion_matrix_test.csv`
  - `outputs/model/run_metadata.json`
  - `outputs/model/run_log.txt`
- Verification performed: YES
- Repeated-run check performed: YES

## 5. Methodological checks
- Test-set misuse: NO
- Leakage risk found: NO
- Label handling correct: YES
- Validation-only threshold/model decision preserved: YES
- Split rule preserved: YES
- Random seed documented and applied: YES

## 6. Failures or unresolved issues
- No fatal execution failures occurred in the notebook baseline pipeline.

## 7. Assumptions
- `train.csv` and `test.csv` are treated as the original Kaggle source files rather than pre-saved benchmark split artefacts.
- The benchmark split was reconstructed by combining the two CSV files and applying a stratified 70/15/15 split with random seed `42`.
- Identifier-style columns (`Unnamed: 0`, `id`) were excluded from modelling to avoid obvious non-predictive leakage-like signals.

## 8. Not executed or unverified items
- Alternative threshold tuning beyond the default 0.5: NOT EXECUTED

## 9. Files created or reviewed
- `Untitled-1.ipynb`
- `docs/reproducibility_record.md`
- `outputs/model/metrics_validation.json`
- `outputs/model/metrics_test.json`
- `outputs/model/validation_predictions.csv`
- `outputs/model/test_predictions.csv`
- `outputs/model/model.joblib`
- `outputs/model/split_manifest.json`
- `outputs/model/feature_manifest.json`
- `outputs/model/confusion_matrix_validation.csv`
- `outputs/model/confusion_matrix_test.csv`
- `outputs/model/run_metadata.json`
- `outputs/model/run_log.txt`
- `outputs/model/repeat_run_check.json`
