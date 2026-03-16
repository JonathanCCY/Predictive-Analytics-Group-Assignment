# Decision Log

Records design decisions made during the benchmark workflow, with rationale.

---

## Step 2 — EDA and Insight Generation

### D-001: Unnamed column identified as `Unnamed: 0`
- **Decision**: Drop the first column which pandas reads as `Unnamed: 0`
- **Rationale**: It is the row index from the original Kaggle export, not a feature
- **Evidence**: Column header is empty in CSV; values are sequential integers

### D-002: Target encoding
- **Decision**: Map `satisfied` → 1, `neutral or dissatisfied` → 0
- **Rationale**: Specified by benchmark prompt; positive class = satisfied

### D-003: Stratified split method
- **Decision**: Two-stage `train_test_split` — first 70/30, then 50/50 on the remainder
- **Rationale**: Produces exact 70/15/15 split with stratification on target. Using `random_state=42` at both stages for determinism.
- **Verification**: Re-run produces identical indices (stability check passed)

### D-004: Missing value handling deferred
- **Decision**: Only `Arrival Delay in Minutes` has missing values (293/90,916 = 0.32%). No imputation performed in EDA.
- **Rationale**: Imputation is a preprocessing step that belongs in the modelling pipeline (Step 3). Median imputation is recommended given the heavy right skew.

### D-005: Arrival Delay flagged as near-duplicate
- **Decision**: Flag `Arrival Delay in Minutes` as a potential redundancy/leakage risk (r = 0.966 with `Departure Delay in Minutes`)
- **Rationale**: The two delay columns carry nearly identical information. Models should handle this gracefully; tree-based models are robust but logistic regression may be affected.
- **Action**: No column removed in EDA; decision deferred to modelling step.

### D-006: Association measures chosen
- **Decision**: Point-biserial |r| for numeric features, Cramer's V for categorical features
- **Rationale**: Standard univariate measures for binary target. Simple, interpretable, and appropriate for initial screening.

### D-007: EDA restricted to training set
- **Decision**: All target-aware statistics and plots computed on training rows only (90,916 rows)
- **Rationale**: Benchmark requirement to prevent information leakage from validation/test into feature decisions

---

## Step 3 — Controlled Baseline Model Training and Comparison

### D-008: Split manifest reused from Step 2
- **Decision**: Loaded `outputs/shared/split_manifest.json` and reconstructed splits by stored indices
- **Rationale**: Benchmark requires canonical split to be created once (Step 2) and reused without modification
- **Verification**: Index counts match manifest; no overlap between splits; manifest file unchanged

### D-009: Model-specific preprocessing pipelines
- **Decision**: Each model uses exactly the preprocessing specified in the prompt:
  - LR: median impute + StandardScaler (numeric), most_frequent + OneHotEncoder (categorical)
  - RF/ET: median impute (numeric), most_frequent + OneHotEncoder (categorical)
  - HGBC: median impute (numeric), most_frequent + OrdinalEncoder (categorical)
- **Rationale**: Fixed by benchmark specification. Different preprocessors reflect model requirements (LR needs scaling; HGBC handles ordinals natively).

### D-010: Model selection — HistGradientBoostingClassifier
- **Decision**: Selected HistGradientBoostingClassifier as final model
- **Rationale**: Highest validation ROC-AUC (0.994632) among all 4 candidates. No tie-breaking needed.
- **Ranking**: HGBC (0.9946) > RF (0.9937) > ET (0.9931) > LR (0.9266)

### D-011: No threshold tuning performed
- **Decision**: Classification threshold fixed at 0.5
- **Rationale**: No pre-defined threshold tuning rule was declared before viewing results, so the default 0.5 is used per benchmark specification.

### D-012: Test metrics labelled post-selection descriptive
- **Decision**: Test metrics for all 4 models are computed and saved but explicitly labelled "post-selection descriptive comparison"
- **Rationale**: Benchmark requires test metrics not be used for model selection. They are reported for completeness after the validation-governed selection.

### D-013: All features retained for modelling
- **Decision**: All 18 numeric and 4 categorical features used. No feature removal despite redundancy between delay columns.
- **Rationale**: Benchmark specifies fixed preprocessing — no feature selection step. Tree-based models handle redundancy naturally; LR is not the selected model.

---

## Step 4 — Debug Broken Pipeline

### D-014: Bug inventory and classification
- **Decision**: Identified 15 distinct bugs in `broken_pipeline.py`, classified as:
  - **Coding bugs** (3): B9 (fit on wrong labels — crash), B10 (misleading "Test" label on val report), B12 (pickle text mode — crash), B14 (wrong output path)
  - **Methodological flaws** (5): B1 (identifier columns not dropped), B2 (preprocessor fit on entire dataset), B5 (split after preprocessing), B11 (test set never evaluated), B15 (no metrics saved)
  - **Reproducibility defects** (5): B3 (no random_state on splits), B4 (canonical split not reused), B6 (max_iter=500 not 1000), B7 (no random_state on LR), B8 (solver not explicit), B13 (pickle instead of joblib)
  - Some bugs span multiple categories (B10 is both coding bug and methodological flaw)
- **Evidence**: CONFIRMED_BY_FILE_INSPECTION of `broken_pipeline.py`

### D-015: Two fatal crashes identified
- **Decision**: B9 (line 59) crashes with `ValueError` due to shape mismatch (X_train has 90,916 rows but y_temp has 38,964 rows). B12 (line 70) would crash with `TypeError` if reached.
- **Evidence**: CONFIRMED_BY_FILE_INSPECTION — `model.fit(X_train, y_temp)` where X_train comes from 70% split but y_temp is the 30% remainder labels.

### D-016: Canonical split reused in fix
- **Decision**: Replaced both `train_test_split` calls with index reconstruction from `outputs/shared/split_manifest.json`
- **Rationale**: Benchmark requires canonical split from Step 2 to be reused without regeneration.
- **Verification**: Split indices verified against manifest; metrics match Step 3 LR exactly.

### D-017: Minimum-change correction policy
- **Decision**: Preserved original LogisticRegression intent. Changed only what was broken or non-compliant.
- **Changes**: Added column drops, fixed split logic, corrected fit target, added random_state/solver/max_iter, replaced pickle with joblib, added test evaluation, added metric saving.
- **Preserved**: Same preprocessing structure (median impute + StandardScaler for numeric, most_frequent + OneHotEncoder for categorical), same model class.

### D-018: Cross-validation against Step 3
- **Decision**: Verified that debug LR metrics match Step 3's LogisticRegression metrics exactly
- **Evidence**: Val ROC-AUC=0.926597, Test ROC-AUC=0.927669 — identical in both steps. CONFIRMED_BY_EXECUTION.

---

## Step 5 — Documentation

### D-019: Evidence-only documentation policy
- **Decision**: All metrics, parameters, and claims in documentation sourced from locally inspected files only
- **Rationale**: Benchmark specification requires grounding in verifiable evidence. No runtime, token usage, or cost was available from system logs.
- **Action**: Runtime, token usage, and monetary cost written as `UNKNOWN` in all documents.

### D-020: Model card deployment readiness
- **Decision**: Model explicitly stated as "Not production-ready"
- **Rationale**: No fairness assessment, temporal validation, or robustness testing performed. Conservative posture per benchmark requirements.

### D-021: Benchmark summary scope
- **Decision**: Benchmark summary includes artefact inventory for all 4 steps, reproducibility checks, bug inventory, and missing-evidence checklist
- **Rationale**: Designed to be comparison-friendly across agents. A human evaluator can quickly verify completeness and spot-check evidence levels.
