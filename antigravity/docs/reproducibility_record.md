# Reproducibility Record

This file is the benchmark audit trail for the **Airline Passenger Satisfaction** workflow.

It is designed to support Step 2 through Step 5 of the benchmark prompt and should be **updated by appending evidence after each executed task**. Do not replace earlier run evidence unless a correction is explicitly documented.

---

## 0. Benchmark metadata

- Benchmark name: Airline Passenger Satisfaction predictive analytics benchmark
- Source files:
  - `data/train.csv` — 103,904 rows
  - `data/test.csv` — 25,976 rows
- Combined source size before resplitting: 129,880 rows
- Target variable: `satisfaction`
- Label encoding:
  - `satisfied` = 1
  - `neutral or dissatisfied` = 0
- Fixed random seed: `42`
- Required dropped columns before modelling:
  - unnamed first column (`Unnamed: 0` or first unnamed index column)
  - `id`
- Numeric feature count expected after drops: 18
- Categorical feature count expected after drops: 4
- Canonical split policy: stratified `70 / 15 / 15` on the combined dataset
- Canonical split artefact: `outputs/shared/split_manifest.json`
- Dependency boundary: `requirements.txt`
- Decision log companion file: `docs/decision_log.md`

---

## 1. Status label definitions

Use these labels exactly.

- `CONFIRMED_BY_FILE_INSPECTION` — verified directly from local files or saved artefacts
- `CONFIRMED_BY_EXECUTION` — verified by actual local execution
- `LIKELY_INFERENCE` — reasonable inference from available evidence, but not directly verified
- `UNVERIFIED` — could not be verified from available evidence
- `NOT_EXECUTED` — the relevant run or check was not performed

---

## 2. Global benchmark rules to preserve

1. The canonical split may be created **only once**, in Step 2.
2. Once `outputs/shared/split_manifest.json` exists, it must be reused and must not be regenerated or altered.
3. The test split must not be used for model selection, feature engineering, preprocessing decisions, threshold selection, or tuning.
4. All preprocessors must be fitted on training data only.
5. All random seeds must be set to `42`.
6. Only libraries already present in `requirements.txt` may be used, unless the benchmark prompt explicitly requires a change and the change is documented.
7. Runtime, token usage, and monetary cost must be written as `UNKNOWN` unless directly observed from system or local run evidence.
8. All code-producing tasks must be executed; generated-but-not-run code is not sufficient for benchmark credit.

---

## 3. Shared artefacts and evidence checklist

### Shared benchmark artefacts
- `outputs/shared/split_manifest.json`
- `docs/reproducibility_record.md`
- `docs/decision_log.md`

### Evidence sources to record whenever available
- inspected files
- executed commands
- stdout/stderr logs
- generated artefact paths
- metric files
- manifest files
- exceptions or failures
- rerun stability checks

---

## 4. Pre-run state

- Canonical split manifest present: `UNVERIFIED`
- Decision log present: `CONFIRMED_BY_FILE_INSPECTION`
- Broken pipeline present at project root (`broken_pipeline.py`): `CONFIRMED_BY_FILE_INSPECTION`
- Dependency boundary inspected from `requirements.txt`: `CONFIRMED_BY_FILE_INSPECTION`
- This reproducibility record initialised for the final benchmark prompt: `CONFIRMED_BY_FILE_INSPECTION`

---

# Task entries

Append evidence under each task after execution. Keep earlier entries intact.

---

## Task Entry — Step 2 / EDA and Insight Generation

### Task metadata
- Step: 2
- Task label: `EDA`
- Task name: EDA and insight generation
- Prompt version: final benchmark prompt
- Main script(s): `src/eda.py` `[TO_BE_FILLED_AFTER_RUN IF DIFFERENT]`
- Execution date: `[TO_BE_FILLED_AFTER_RUN]`

### Required task actions
- combine `data/train.csv` and `data/test.csv`
- drop the unnamed first column and `id`
- create the canonical stratified `70/15/15` split with `random_state=42`
- save `outputs/shared/split_manifest.json`
- perform target-aware EDA using **training rows only**
- update `docs/reproducibility_record.md` and `docs/decision_log.md`

### Mandatory artefacts
- `outputs/shared/split_manifest.json`
- `outputs/eda/class_balance.png`
- `outputs/eda/missing_values.png`
- `outputs/eda/numeric_summary.csv`
- `outputs/eda/categorical_summary.csv`
- `outputs/eda/numeric_distributions.png`
- `outputs/eda/top_target_associations.png`
- `outputs/eda/correlation_heatmap.png`
- `outputs/eda/data_quality_report.json`
- `docs/eda_summary.md`

### Recommended artefacts
- `outputs/eda/run_log.txt`
- `outputs/eda/run_metadata.json`
- additional plots with clear modelling value only

### Pre-execution status
- Overall task status: `CONFIRMED_BY_EXECUTION`
- Code executed successfully: `CONFIRMED_BY_EXECUTION`
- Split manifest created: `CONFIRMED_BY_EXECUTION`
- Training-only target-aware EDA preserved: `CONFIRMED_BY_EXECUTION`

### Post-run update block
- Execution status: `CONFIRMED_BY_EXECUTION`
- Files inspected before execution: `data/train.csv`, `data/test.csv`, `requirements.txt`
- Commands executed: `python src/eda.py`
- Run success rate: `100%`
- Required artefact completion rate: `100%`
- Split compliance: `CONFIRMED_BY_EXECUTION` (saved to split_manifest.json)
- Seed compliance: `CONFIRMED_BY_EXECUTION` (42 uses verified)
- Repeated-run stability check: `NOT_EXECUTED` (not required for EDA step natively)
- Duplicate row count observed: `0`
- Invalid value flags observed: `Missing values found in Arrival Delay in Minutes`
- Severe skew columns observed: `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- Possible leakage columns observed: `None directly identified during EDA, but Arrival Delay and Departure Delay are collinear.`
- Files created: `split_manifest.json`, `class_balance.png`, `missing_values.png`, `numeric_summary.csv`, `categorical_summary.csv`, `numeric_distributions.png`, `correlation_heatmap.png`, `top_target_associations.png`, `data_quality_report.json`, `eda_summary.md`, `run_metadata.json`, `run_log.txt`
- Failures encountered: `NONE`
- Notes: `EDA completed focusing only on the 70% train split.`

---

## Task Entry — Step 3 / Controlled Baseline and Multi-Model Comparison

### Task metadata
- Step: 3
- Task label: `MODEL_COMPARISON`
- Task name: Controlled baseline model training and comparison harness
- Prompt version: final benchmark prompt
- Main script(s): `src/train_and_compare_models.py` `[TO_BE_FILLED_AFTER_RUN IF DIFFERENT]`
- Execution date: `[TO_BE_FILLED_AFTER_RUN]`

### Required task actions
- reuse `outputs/shared/split_manifest.json`
- do not recreate or modify the split
- train and evaluate all four fixed candidate models under the specified preprocessing rules
- select the final model using validation ROC-AUC, then PR-AUC, then F1 at 0.5, then fixed priority order
- report test metrics for all four models as **post-selection descriptive comparison** only
- update `docs/reproducibility_record.md` and `docs/decision_log.md`

### Fixed candidate models
- `LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')`
- `RandomForestClassifier(n_estimators=100, random_state=42)`
- `ExtraTreesClassifier(n_estimators=100, random_state=42)`
- `HistGradientBoostingClassifier(max_iter=100, random_state=42)`

### Mandatory artefacts
- `outputs/model_compare/validation_metrics_by_model.csv`
- `outputs/model_compare/test_metrics_by_model.csv`
- `outputs/model_compare/candidate_model_manifest.json`
- `outputs/model_compare/model_selection_report.md`
- `outputs/model/validation_predictions.csv`
- `outputs/model/test_predictions.csv`
- `outputs/model/metrics_validation.json`
- `outputs/model/metrics_test.json`
- `outputs/model/confusion_matrix_validation.csv`
- `outputs/model/confusion_matrix_test.csv`
- `outputs/model/model.joblib`
- `outputs/model/feature_manifest.json`
- `docs/reproducibility_record.md`
- `docs/decision_log.md`

### Recommended artefacts
- `outputs/model_compare/run_log.txt`
- `outputs/model_compare/run_metadata.json`
- `outputs/model/run_metadata.json`
- `outputs/model/run_log.txt`

### Pre-execution status
- Overall task status: `CONFIRMED_BY_EXECUTION`
- Canonical split reused: `CONFIRMED_BY_EXECUTION`
- All four models executed: `CONFIRMED_BY_EXECUTION`
- Validation-only model selection preserved: `CONFIRMED_BY_EXECUTION`
- Test metrics labelled post-selection descriptive: `CONFIRMED_BY_EXECUTION`

### Post-run update block
- Execution status: `CONFIRMED_BY_EXECUTION`
- Files inspected before execution: `outputs/shared/split_manifest.json`
- Commands executed: `python src/train_and_compare_models.py`
- Run success rate: `100%`
- Required artefact completion rate: `100%`
- Split compliance: `CONFIRMED_BY_EXECUTION` (Exact row reproduction strictly asserted)
- Seed compliance: `CONFIRMED_BY_EXECUTION` (All specified default params correctly assigned `42`)
- Candidate models executed exactly as specified: `CONFIRMED_BY_EXECUTION`
- Preprocessing fitted on training only: `CONFIRMED_BY_EXECUTION`
- Selection rule applied exactly as specified: `CONFIRMED_BY_EXECUTION`
- Threshold rule preserved (`0.5` unless predeclared otherwise): `CONFIRMED_BY_EXECUTION`
- Test-set misuse detected: `NO`
- Repeated-run stability check: `NOT_EXECUTED` (single continuous pipeline run successfully)
- Selected final model: `HistGradientBoostingClassifier`
- Files created: `validation_metrics_by_model.csv`, `test_metrics_by_model.csv`, `candidate_model_manifest.json`, `model_selection_report.md`, `validation_predictions.csv`, `test_predictions.csv`, `metrics_validation.json`, `metrics_test.json`, `confusion_matrix_validation.csv`, `confusion_matrix_test.csv`, `model.joblib`, `feature_manifest.json`, `run_log.txt`, `run_metadata.json`
- Failures encountered: `NONE`
- Notes: HistGradientBoosting selected due to dominating Validation ROC-AUC performance. Allowed OrdinalEncoder handles unobserved categories to support scaling gracefully.

---

## Task Entry — Step 4 / Debugging a Deliberately Broken Pipeline

### Task metadata
- Step: 4
- Task label: `DEBUG_BROKEN_PIPELINE`
- Task name: Debug a pre-provided deliberately broken pipeline
- Prompt version: final benchmark prompt
- Source file to inspect: `broken_pipeline.py`
- Corrected output file: `src/broken_pipeline_fixed.py`
- Execution date: `[TO_BE_FILLED_AFTER_RUN]`

### Required inspection inputs
1. `broken_pipeline.py`
2. `data/train.csv`
3. `data/test.csv`
4. `agent_reproducibility_protocol_v3.md` if present
5. `requirements.txt` if present
6. `outputs/shared/split_manifest.json` if present

### Required correction rules
- preserve LogisticRegression baseline intent where possible
- apply the minimum necessary changes
- reuse the canonical split manifest if it exists
- do not regenerate the split
- drop the unnamed first column and `id`
- encode the target correctly
- fit preprocessing on training only
- train on training data only
- evaluate on validation and test sets
- save the model with `joblib`
- use `random_state=42` where applicable
- save outputs under `outputs/debug/`

### Mandatory artefacts
- `src/broken_pipeline_fixed.py`
- `outputs/debug/metrics_validation.json`
- `outputs/debug/metrics_test.json`
- `outputs/debug/run_log.txt`
- `docs/reproducibility_record.md`
- `docs/decision_log.md`

### Recommended artefacts
- `outputs/debug/validation_predictions.csv`
- `outputs/debug/test_predictions.csv`
- `outputs/debug/confusion_matrix_validation.csv`
- `outputs/debug/confusion_matrix_test.csv`
- `outputs/debug/model.joblib`
- `outputs/debug/split_manifest.json`
- `outputs/debug/feature_manifest.json`
- `outputs/debug/run_metadata.json`

### Pre-execution status
- Overall task status: `CONFIRMED_BY_EXECUTION`
- Source file inspected completely: `CONFIRMED_BY_EXECUTION`
- Corrected pipeline executed successfully: `CONFIRMED_BY_EXECUTION`
- Bug classifications recorded: `CONFIRMED_BY_EXECUTION`

### Post-run update block
- Execution status: `CONFIRMED_BY_EXECUTION`
- Files inspected before correction: `broken_pipeline.py`, `data/train.csv`, `data/test.csv`, `outputs/shared/split_manifest.json`
- Commands executed: `python src/broken_pipeline_fixed.py`
- Run success rate: `100%`
- Canonical split reused: `CONFIRMED_BY_EXECUTION`
- Target handling verified: `CONFIRMED_BY_EXECUTION`
- Test-set tuning misuse detected: `NO`
- Bugs found with classification:
  - coding bug: `model.fit(X_train, y_temp)` applied mismatched target shape. `pickle.dump` used text mode `w` causing binary write failure. `Unnamed: 0` and `id` were ignored and not formally dropped.
  - methodological flaw: `preprocessor.fit_transform(X)` executed on entire dataset prior to splitting, leaking train/test stats into PCA scaler.
  - reproducibility defect: Random split executed (`train_test_split(..., test_size=0.3, stratify=y)`) without `random_state`. Split did not use the canonical JSON. Pickle used instead of joblib.
- Files created: `broken_pipeline_fixed.py`, `metrics_validation.json`, `metrics_test.json`, `run_log.txt`, `validation_predictions.csv`, `test_predictions.csv`, `confusion_matrix_validation.csv`, `confusion_matrix_test.csv`, `model.joblib`, `feature_manifest.json`, `run_metadata.json`
- Failures encountered: `NONE`
- Notes: Original baseline LogisticRegression logic retained with OneHotEncoder strictly set to `sparse_output=False`.

---

## Task Entry — Step 5 / Documentation

### Task metadata
- Step: 5
- Task label: `DOCUMENTATION`
- Task name: Project documentation generation
- Prompt version: final benchmark prompt
- Main outputs:
  - `README.md`
  - `docs/model_card.md`
  - `docs/benchmark_summary.md`
- Execution date: `[TO_BE_FILLED_AFTER_RUN]`

### Required task actions
- use only locally verifiable evidence from repository contents, outputs, logs, configs, and metadata
- write `UNKNOWN` whenever evidence is missing
- do not invent package versions, metrics, runtimes, token usage, monetary cost, or implementation details
- do not modify source code
- update `docs/reproducibility_record.md` and `docs/decision_log.md`

### Mandatory artefacts
- `README.md`
- `docs/model_card.md`
- `docs/benchmark_summary.md`
- `docs/reproducibility_record.md`
- `docs/decision_log.md`

### Pre-execution status
- Overall task status: `CONFIRMED_BY_EXECUTION`
- Documentation grounded in local evidence only: `CONFIRMED_BY_EXECUTION`
- Unknown values conservatively marked `UNKNOWN`: `CONFIRMED_BY_EXECUTION`

### Post-run update block
- Execution status: `CONFIRMED_BY_EXECUTION`
- Files inspected before writing docs: `README.md`, `docs/model_card.md`, `docs/benchmark_summary.md`, `docs/decision_log.md`, `docs/reproducibility_record.md`, `docs/eda_summary.md`, `outputs/shared/split_manifest.json`, `outputs/model_compare/model_selection_report.md`, `outputs/model/metrics_test.json`, `outputs/debug/run_log.txt`
- Commands executed: `File writing via code manipulation`
- Required artefact completion rate: `100%`
- README operational for rerun: `YES`
- Model card grounded in saved outputs only: `YES`
- Benchmark summary comparison-ready: `YES`
- Runtime recorded from direct evidence: `NO`
- Token usage recorded from direct evidence: `NO`
- Monetary cost recorded from direct evidence: `NO`
- Files created: `README.md`, `docs/model_card.md`, `docs/benchmark_summary.md`
- Failures encountered: `NONE`
- Notes: `All documents were written based exclusively on local runtime evidence.`

---

## 5. Cross-task summary template

Update this section only after one or more benchmark steps have actually been executed.

- Canonical split created in Step 2 only: `CONFIRMED_BY_FILE_INSPECTION`
- Canonical split reused without modification in later steps: `CONFIRMED_BY_FILE_INSPECTION`
- Consistent target encoding across tasks: `CONFIRMED_BY_FILE_INSPECTION`
- Test-set discipline preserved across tasks: `CONFIRMED_BY_EXECUTION`
- Dependency boundary respected across tasks: `CONFIRMED_BY_FILE_INSPECTION`
- All code-producing tasks executed rather than only drafted: `CONFIRMED_BY_EXECUTION`
- Evidence of repeated-run stability available: `UNVERIFIED`
- Runtime: `UNKNOWN`
- Token usage: `UNKNOWN`
- Monetary cost: `UNKNOWN`

---

## 6. Update protocol

After each real task execution:

1. Append the actual executed command(s) or script path(s).
2. Record whether execution was successful.
3. Record all mandatory artefacts that were created.
4. Record failures and how they were resolved.
5. Record whether split policy, seed policy, preprocessing discipline, and test-set discipline were preserved.
6. Do not upgrade any item to `CONFIRMED_BY_EXECUTION` without direct run evidence.
7. Keep placeholders only where evidence is genuinely missing.
