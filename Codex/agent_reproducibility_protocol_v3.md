# Agent Reproducibility Protocol

## Purpose

This file defines the mandatory reproducibility and auditability rules for all agents participating in the predictive analytics benchmarking workflow.

The goal is not only to assess one-off output quality, but also to assess whether an agent can generate a transparent, rerunnable, methodologically coherent, and auditable workflow under fixed task conditions.

This protocol must be followed for every task unless a task explicitly overrides a specific requirement.

---

## Benchmark Fairness Rules

To make cross-agent comparison fair and interpretable, all agents must be evaluated under the same local task conditions.

Unless a task explicitly permits otherwise:

- use only the local project files made available by the evaluator
- do not rely on hidden files, prior session state, or external connectors
- do not browse the web or use external data sources
- do not invent outputs, files, execution results, package versions, or citations
- do not assume that notebook content is a binding implementation reference unless the task explicitly instructs this

If a capability is unavailable, state the limitation explicitly rather than silently substituting another workflow.

---

## Project Context

- Dataset: Airline Passenger Satisfaction
- Prediction task: supervised binary classification
- Target variable: `satisfaction`
- Label encoding: `satisfied = 1`, `neutral or dissatisfied = 0`
- Data policy: local files only
- Split policy: custom stratified 70/15/15 train/validation/test split
- Test-set restriction: the test set must not be used for feature engineering, model selection, threshold tuning, or any modelling decision
- Random seed: `42`
- Local workflow expectation: standard Python execution by a human evaluator

---

## Core Principle

Reproducibility must be **evidenced, not asserted**.

Do not claim that something is reproducible, verified, stable, or successfully executed unless that claim is supported by explicit execution evidence or clearly scoped verification.

Whenever uncertainty exists, state it explicitly.

---

## Required Reproducibility Deliverable

For every task, create or update the following file:

- `docs/reproducibility_record.md`

When runnable code is produced, also create or update the project-level dependency file when relevant:

- `requirements.txt`

This file is the main audit log for reproducibility and must be maintained even if execution is partial, fails, or is not performed.

Each task entry inside `docs/reproducibility_record.md` must include a unique task label, for example:

- `EDA`
- `BASELINE_MODEL`
- `DEBUG_BROKEN_PIPELINE`
- `DOCUMENTATION`
- `CODE_QUALITY`

If multiple tasks are completed in the same workspace, append a new section rather than overwriting the previous one.

---

## Four Mandatory Quantitative Reproducibility Indicators

These four indicators must be reported whenever code, outputs, or pipeline artefacts are involved.

### 1. Run Success Rate

**Definition**: the proportion of attempted runs that complete without fatal error under the same task conditions.

**Format**:
- `successful_runs / attempted_runs`
- Example: `2 / 3`

**Rules**:
- Count only genuine execution attempts.
- If no runs were attempted, record: `not executed`.
- Do not infer success from code generation alone.

---

### 2. Required Artefact Completion Rate

**Definition**: the proportion of required output files that were created and are non-empty.

**Format**:
- `created_required_files / total_required_files`
- Example: `7 / 9`

**Rules**:
- Compare against the task-specific required output list.
- A file counts as complete only if it exists and is non-empty.
- If output verification was not performed, record: `unverified`.

---

### 3. Split and Seed Compliance

**Definition**: whether the pipeline preserved the required split logic and fixed random seed.

**Report separately**:
- `split compliance: PASS / FAIL / UNVERIFIED / NOT APPLICABLE`
- `seed compliance: PASS / FAIL / UNVERIFIED / NOT APPLICABLE`

**Rules**:
- Split compliance requires preserving the custom stratified 70/15/15 split rule.
- Seed compliance requires using and documenting `42` as the fixed random seed.
- If the task is non-modelling and this is not applicable, record: `NOT APPLICABLE`.

---

### 4. Output Stability Across Repeated Runs

**Definition**: whether repeated runs under the same settings produce identical or near-identical outputs.

**At minimum compare, where relevant**:
- evaluation metrics
- prediction outputs
- key coefficients or model parameters
- confusion matrices, if such files are produced

**Suggested report fields**:
- `metrics stable across repeated runs: YES / NO / NOT EXECUTED / NOT APPLICABLE`
- `prediction outputs identical across repeated runs: YES / NO / NOT EXECUTED / NOT APPLICABLE`
- `max ROC-AUC difference across runs: <value or not executed>`
- `max F1 difference across runs: <value or not executed>`

**Rules**:
- Do not claim output stability unless repeated execution was actually performed.
- If repeated execution was not performed, write: `NOT EXECUTED`.
- For deterministic tabular pipelines in this benchmark, identical outputs are the default expectation unless a justified non-deterministic source is explicitly documented.

---

## Fixed Repeated-Run Rule for Executable Tasks

To reduce cross-agent ambiguity, use the following repeated-run rule:

- For executable modelling or debugging tasks: attempt **3 runs** under the same settings when execution is feasible.
- For EDA tasks with generated scripts: at least **1 confirmed run** is required; repeated runs are optional unless the evaluator explicitly asks for them.
- For documentation-only or code-quality-only tasks: repeated execution is **NOT APPLICABLE** unless runnable code is also created.

If 3 runs are not attempted for an executable modelling or debugging task, record the reason explicitly in `docs/reproducibility_record.md`.

---

## Required Status Labels

Every claim inside `docs/reproducibility_record.md` must fall into one of the following categories:

- **CONFIRMED BY EXECUTION**
- **CREATED BUT NOT EXECUTED**
- **PARTIALLY EXECUTED**
- **FAILED DURING EXECUTION**
- **EXPECTED BUT UNVERIFIED**
- **NOT EXECUTED**
- **NOT APPLICABLE**

Do not blur these categories.

---

## Methodological Violation Logging

The reproducibility record must explicitly log whether any of the following occurred:

- test-set misuse
- data leakage
- incorrect label handling
- incorrect or undocumented split logic
- missing fixed seed
- unstable or undocumented non-deterministic behaviour
- missing required artefacts
- manual intervention needed to make code run

For each issue, record one of:
- `YES`
- `NO`
- `UNVERIFIED`

If `YES`, add a brief explanation.

Major methodological violations must also be highlighted in the task summary.

---

## Mandatory Behaviour Rules

### Rule 1. Do not fabricate execution

Generating code is **not** the same as executing code.

If a script was written but not run, state:
- `CREATED BUT NOT EXECUTED`

### Rule 2. Do not fabricate outputs

If a file was expected but not confirmed, state:
- `EXPECTED BUT UNVERIFIED`

### Rule 3. Do not fabricate stability

If repeated runs were not attempted, state:
- `output stability: NOT EXECUTED`

### Rule 4. Distinguish assumptions from evidence

Every assumption must be separated from observed execution evidence.

### Rule 5. Preserve test discipline

No modelling decision may use the test set.

### Rule 6. Preserve split discipline

If the split is generated during the task, it must follow the custom stratified 70/15/15 rule.

### Rule 7. Preserve seed discipline

Use fixed random seed `42` whenever randomness is relevant.

### Rule 8. Keep the log concise and comparable

The record should be compact, structured, and suitable for side-by-side comparison across agents.

### Rule 9. Record manual intervention explicitly

If the evaluator or user had to manually change paths, install packages, edit files, or rerun commands to make the workflow succeed, record that intervention.

---

## Suggested Support Files for Runnable Tasks

When a task includes runnable code, create reproducibility-supporting files where feasible.

Preferred files:

- `src/repeat_run_check.py`
- `src/verify_outputs.py`
- `tests/test_pipeline_outputs.py`

These are strongly recommended for modelling, debugging, and other executable pipeline tasks.

### Expected purpose of each file

#### `src/repeat_run_check.py`
A lightweight harness that:
- reruns the main pipeline multiple times under the same conditions
- records success or failure of each run
- compares required artefacts across runs
- compares key metrics and predictions across runs
- summarises repeated-run consistency

#### `src/verify_outputs.py`
A verification script that:
- checks required files exist
- checks files are non-empty
- optionally verifies expected columns or metric keys
- optionally writes a concise verification summary

#### `tests/test_pipeline_outputs.py`
A lightweight test module that can include smoke tests such as:
- target variable exists
- label mapping is handled correctly
- split files or split manifest exist when required
- metric JSON files contain required keys
- prediction files contain expected columns
- no prohibited test-set usage is encoded in the workflow logic

---

## Automation Guidance

### What can be automated

The following can be automated by the agent through generated local scripts:

- pipeline execution
- repeated-run checking
- artefact verification
- basic smoke tests
- summary generation for reproducibility logging

### What must not be assumed

Do not assume that automated checks were actually run unless execution evidence exists.

If automation scripts are created but not executed, state this clearly.

### Preferred workflow for runnable tasks

1. generate the main pipeline
2. generate verification and repeated-run support scripts
3. run the main pipeline if execution is allowed
4. run verification scripts if execution is allowed
5. run repeated-run checks if execution is allowed and applicable
6. update `docs/reproducibility_record.md`
7. separate confirmed evidence from non-executed items

---

## Task-Specific Expectations

### A. EDA Tasks

For EDA tasks:
- create the required plots and summary files
- verify that required EDA artefacts exist
- log whether outputs were generated and verified
- if no code was executed, do not imply that plots or summaries exist

Recommended reproducibility checks:
- required artefact completion rate
- run success rate if scripts were executed
- split compliance if EDA relied on pre-split training data

---

### B. Baseline Modelling Tasks

For baseline modelling tasks:
- preserve the custom stratified 70/15/15 rule
- preserve seed = 42
- use validation only for threshold or model-selection decisions
- keep the test set untouched until final evaluation
- save the task-required model outputs
- perform 3 repeated runs when feasible
- log output stability if repeated runs were performed

Minimum required baseline modelling artefacts:
- `outputs/model/metrics_validation.json`
- `outputs/model/metrics_test.json`
- `outputs/model/validation_predictions.csv`
- `outputs/model/test_predictions.csv`
- `outputs/model/model.joblib`
- `outputs/model/split_manifest.json`
- `outputs/model/feature_manifest.json`

Recommended additional audit artefacts:
- `outputs/model/confusion_matrix_validation.csv`
- `outputs/model/confusion_matrix_test.csv`
- `outputs/model/run_metadata.json`
- `outputs/model/run_log.txt`
- `outputs/model/split_assignments.csv`

Recommended supporting files:
- `src/repeat_run_check.py`
- `src/verify_outputs.py`
- `tests/test_pipeline_outputs.py`

---

### C. Debugging Tasks

For debugging tasks:
- identify root cause
- distinguish coding bugs from methodological flaws
- verify that the corrected version preserves split and test discipline
- log whether the corrected pipeline was executed end to end
- attempt 3 repeated runs if the corrected pipeline is executable and the task is intended to be rerunnable
- record any remaining reproducibility risks

---

### D. Documentation Tasks

For documentation tasks:
- do not fabricate runtime results, metric values, versions, or execution status
- write `UNKNOWN`, `[TO BE FILLED AFTER RUN]`, `NOT EXECUTED`, or `UNVERIFIED` where evidence is missing
- documentation must match available evidence only
- `docs/reproducibility_record.md` should still be updated to reflect what was written versus what was actually verified

---

### E. Code Quality Reporting Tasks

For code quality tasks:
- do not fabricate measured quality metrics that require execution or external tools
- label external metrics as `To be filled by evaluator` where appropriate
- distinguish direct evidence from expected behaviour
- include reproducibility risks as a separate subsection

---

## Required Structure for `docs/reproducibility_record.md`

Use the following structure unless a task explicitly overrides it.

# Reproducibility Record

## 1. Task
- Task label:
- Task name:
- Date:
- Main script(s):

## 2. Execution status
- Status:
- Environment notes:
- Manual intervention required: YES / NO / UNVERIFIED

## 3. Quantitative indicators
- Run success rate:
- Required artefact completion rate:
- Split compliance:
- Seed compliance:
- Output stability across repeated runs:

## 4. Evidence
- Attempted runs:
- Successful runs:
- Required files expected:
- Required files created:
- Verification performed: YES / NO
- Repeated-run check performed: YES / NO

## 5. Methodological checks
- Test-set misuse:
- Leakage risk found:
- Label handling correct:
- Validation-only threshold/model decision preserved:
- Split rule preserved:
- Random seed documented and applied:

## 6. Failures or unresolved issues
- 

## 7. Assumptions
- 

## 8. Not executed or unverified items
- 

## 9. Files created or reviewed
- 

---

## Preferred Prediction File Columns

When predictions are saved, use the following columns where applicable:

- `row_id`
- `id` (if available in the source data)
- `y_true`
- `y_prob`
- `y_pred`

This improves auditability and repeated-run comparison.

---

## Preferred Modelling Audit Artefacts

When a modelling pipeline creates or applies the project split, use the following structure unless a task explicitly overrides it.

Minimum artefacts:
- `outputs/model/metrics_validation.json`
- `outputs/model/metrics_test.json`
- `outputs/model/validation_predictions.csv`
- `outputs/model/test_predictions.csv`
- `outputs/model/model.joblib`
- `outputs/model/split_manifest.json`
- `outputs/model/feature_manifest.json`

Recommended additional artefacts:
- `outputs/model/confusion_matrix_validation.csv`
- `outputs/model/confusion_matrix_test.csv`
- `outputs/model/run_metadata.json`
- `outputs/model/run_log.txt`
- `outputs/model/split_assignments.csv`

These files strengthen reproducibility evidence and support cross-agent comparison, but the recommended artefacts should not be counted as missing required files unless the task explicitly makes them mandatory.

---

## Minimum Expectations for Verification Scripts

### `src/verify_outputs.py`
Should, where applicable:
- verify all task-required output files exist
- verify files are non-empty
- verify required metric keys exist in JSON files
- verify prediction files contain expected columns
- verify confusion matrix files are readable if such files are produced
- optionally write a short verification summary

### `src/repeat_run_check.py`
Should, where applicable:
- rerun the task under the same settings **3 times** for executable modelling or debugging tasks unless explicitly exempted
- record run-level success or failure
- compare selected metrics across runs
- compare prediction files across runs
- report whether outputs are identical, near-identical, or inconsistent
- avoid overwriting prior evidence without explicit logging

### `tests/test_pipeline_outputs.py`
Should remain lightweight and test local correctness assumptions only.
Do not create bloated or framework-heavy tests unless justified by the task.

---

## Scoring-Friendly Logging Guidance

To make cross-agent comparison easier, keep entries compact and standardised.

Preferred conventions:
- use `PASS / FAIL / UNVERIFIED / NOT EXECUTED / NOT APPLICABLE`
- use explicit ratios for quantitative indicators
- list artefacts with relative paths
- separate observed evidence from assumptions
- avoid vague phrases such as “should work”, “appears reproducible”, or “likely stable” unless clearly labelled as unverified judgement

---

## Final Instruction to the Agent

Follow this protocol as a binding local reproducibility and auditability specification.

If a task is executable, prefer to generate the main pipeline plus lightweight verification support files.

If execution cannot be confirmed, make that limitation explicit.

Never turn absence of evidence into evidence of success.
