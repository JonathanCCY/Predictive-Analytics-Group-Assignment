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
- Decision log present: `UNVERIFIED`
- Broken pipeline present at project root (`broken_pipeline.py`): `UNVERIFIED`
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
- Overall task status: `NOT_EXECUTED`
- Code executed successfully: `NOT_EXECUTED`
- Split manifest created: `NOT_EXECUTED`
- Training-only target-aware EDA preserved: `UNVERIFIED`

### Post-run update block
- Execution status: `[NOT_EXECUTED | CONFIRMED_BY_EXECUTION | UNVERIFIED]`
- Files inspected before execution: `[TO_BE_FILLED_AFTER_RUN]`
- Commands executed: `[TO_BE_FILLED_AFTER_RUN]`
- Run success rate: `[TO_BE_FILLED_AFTER_RUN]`
- Required artefact completion rate: `[TO_BE_FILLED_AFTER_RUN]`
- Split compliance: `[TO_BE_FILLED_AFTER_RUN]`
- Seed compliance: `[TO_BE_FILLED_AFTER_RUN]`
- Repeated-run stability check: `[TO_BE_FILLED_AFTER_RUN]`
- Duplicate row count observed: `[TO_BE_FILLED_AFTER_RUN]`
- Invalid value flags observed: `[TO_BE_FILLED_AFTER_RUN]`
- Severe skew columns observed: `[TO_BE_FILLED_AFTER_RUN]`
- Possible leakage columns observed: `[TO_BE_FILLED_AFTER_RUN]`
- Files created: `[TO_BE_FILLED_AFTER_RUN]`
- Failures encountered: `[NONE | TO_BE_FILLED_AFTER_RUN]`
- Notes: `[TO_BE_FILLED_AFTER_RUN]`

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
- Overall task status: `NOT_EXECUTED`
- Canonical split reused: `UNVERIFIED`
- All four models executed: `NOT_EXECUTED`
- Validation-only model selection preserved: `UNVERIFIED`
- Test metrics labelled post-selection descriptive: `UNVERIFIED`

### Post-run update block
- Execution status: `[NOT_EXECUTED | CONFIRMED_BY_EXECUTION | UNVERIFIED]`
- Files inspected before execution: `[TO_BE_FILLED_AFTER_RUN]`
- Commands executed: `[TO_BE_FILLED_AFTER_RUN]`
- Run success rate: `[TO_BE_FILLED_AFTER_RUN]`
- Required artefact completion rate: `[TO_BE_FILLED_AFTER_RUN]`
- Split compliance: `[TO_BE_FILLED_AFTER_RUN]`
- Seed compliance: `[TO_BE_FILLED_AFTER_RUN]`
- Candidate models executed exactly as specified: `[TO_BE_FILLED_AFTER_RUN]`
- Preprocessing fitted on training only: `[TO_BE_FILLED_AFTER_RUN]`
- Selection rule applied exactly as specified: `[TO_BE_FILLED_AFTER_RUN]`
- Threshold rule preserved (`0.5` unless predeclared otherwise): `[TO_BE_FILLED_AFTER_RUN]`
- Test-set misuse detected: `[NO | YES | TO_BE_FILLED_AFTER_RUN]`
- Repeated-run stability check: `[TO_BE_FILLED_AFTER_RUN]`
- Selected final model: `[TO_BE_FILLED_AFTER_RUN]`
- Files created: `[TO_BE_FILLED_AFTER_RUN]`
- Failures encountered: `[NONE | TO_BE_FILLED_AFTER_RUN]`
- Notes: `[TO_BE_FILLED_AFTER_RUN]`

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
- Overall task status: `NOT_EXECUTED`
- Source file inspected completely: `NOT_EXECUTED`
- Corrected pipeline executed successfully: `NOT_EXECUTED`
- Bug classifications recorded: `NOT_EXECUTED`

### Post-run update block
- Execution status: `[NOT_EXECUTED | CONFIRMED_BY_EXECUTION | UNVERIFIED]`
- Files inspected before correction: `[TO_BE_FILLED_AFTER_RUN]`
- Commands executed: `[TO_BE_FILLED_AFTER_RUN]`
- Run success rate: `[TO_BE_FILLED_AFTER_RUN]`
- Canonical split reused: `[TO_BE_FILLED_AFTER_RUN]`
- Target handling verified: `[TO_BE_FILLED_AFTER_RUN]`
- Test-set tuning misuse detected: `[NO | YES | TO_BE_FILLED_AFTER_RUN]`
- Bugs found with classification:
  - coding bug: `[TO_BE_FILLED_AFTER_RUN]`
  - methodological flaw: `[TO_BE_FILLED_AFTER_RUN]`
  - reproducibility defect: `[TO_BE_FILLED_AFTER_RUN]`
- Files created: `[TO_BE_FILLED_AFTER_RUN]`
- Failures encountered: `[NONE | TO_BE_FILLED_AFTER_RUN]`
- Notes: `[TO_BE_FILLED_AFTER_RUN]`

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
- Overall task status: `NOT_EXECUTED`
- Documentation grounded in local evidence only: `UNVERIFIED`
- Unknown values conservatively marked `UNKNOWN`: `UNVERIFIED`

### Post-run update block
- Execution status: `[NOT_EXECUTED | CONFIRMED_BY_EXECUTION | UNVERIFIED]`
- Files inspected before writing docs: `[TO_BE_FILLED_AFTER_RUN]`
- Commands executed: `[TO_BE_FILLED_AFTER_RUN]`
- Required artefact completion rate: `[TO_BE_FILLED_AFTER_RUN]`
- README operational for rerun: `[TO_BE_FILLED_AFTER_RUN]`
- Model card grounded in saved outputs only: `[TO_BE_FILLED_AFTER_RUN]`
- Benchmark summary comparison-ready: `[TO_BE_FILLED_AFTER_RUN]`
- Runtime recorded from direct evidence: `[YES | NO]`
- Token usage recorded from direct evidence: `[YES | NO]`
- Monetary cost recorded from direct evidence: `[YES | NO]`
- Files created: `[TO_BE_FILLED_AFTER_RUN]`
- Failures encountered: `[NONE | TO_BE_FILLED_AFTER_RUN]`
- Notes: `[TO_BE_FILLED_AFTER_RUN]`

---

## 5. Cross-task summary template

Update this section only after one or more benchmark steps have actually been executed.

- Canonical split created in Step 2 only: `[TO_BE_FILLED_AFTER_RUN]`
- Canonical split reused without modification in later steps: `[TO_BE_FILLED_AFTER_RUN]`
- Consistent target encoding across tasks: `[TO_BE_FILLED_AFTER_RUN]`
- Test-set discipline preserved across tasks: `[TO_BE_FILLED_AFTER_RUN]`
- Dependency boundary respected across tasks: `[TO_BE_FILLED_AFTER_RUN]`
- All code-producing tasks executed rather than only drafted: `[TO_BE_FILLED_AFTER_RUN]`
- Evidence of repeated-run stability available: `[TO_BE_FILLED_AFTER_RUN]`
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
