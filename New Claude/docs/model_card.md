# Model Card — Airline Passenger Satisfaction Classifier

## Model Purpose

Predict whether an airline passenger is **satisfied** or **neutral/dissatisfied** based on flight details and survey responses. This model is a benchmark artefact produced under controlled conditions for comparative evaluation of LLM-assisted predictive analytics workflows.

## Target Variable

- **Name**: `satisfaction`
- **Encoding**: `satisfied` = 1 (positive class), `neutral or dissatisfied` = 0 (negative class)
- **Task type**: supervised binary classification

## Data

- **Source**: Kaggle Airline Passenger Satisfaction dataset
- **Files**: `data/train.csv` (103,904 rows) + `data/test.csv` (25,976 rows)
- **Combined**: 129,880 rows, 23 columns after dropping `Unnamed: 0` and `id`
- **Features**: 18 numeric + 4 categorical = 22 input features
- **Missing values**: `Arrival Delay in Minutes` only (293 rows in training set, 0.32%)
- **Class balance (training set)**: 56.6% neutral/dissatisfied, 43.4% satisfied

## Split Policy

**Intended**: Stratified 70/15/15 (train/validation/test) with `random_state=42`
**Actual** (from `outputs/shared/split_manifest.json`):

| Split | Rows | Class 0 | Class 1 | Satisfied rate |
|-------|------|---------|---------|----------------|
| Train | 90,916 | 51,416 | 39,500 | 43.45% |
| Validation | 19,482 | 11,018 | 8,464 | 43.45% |
| Test | 19,482 | 11,018 | 8,464 | 43.45% |

**Source**: CONFIRMED_BY_FILE_INSPECTION of `outputs/shared/split_manifest.json`

### Test Set Restrictions

The test set was **not** used for:
- Model selection
- Hyperparameter tuning
- Threshold selection
- Feature engineering or preprocessing decisions

Test metrics are reported as **post-selection descriptive** only, after the final model was chosen using validation metrics.

## Candidate Models and Selection

### Candidates

| Model | Key Parameters | Preprocessing |
|-------|---------------|---------------|
| LogisticRegression | max_iter=1000, solver=lbfgs, random_state=42 | Median impute + StandardScaler (numeric); most_frequent impute + OneHotEncoder (categorical) |
| RandomForestClassifier | n_estimators=100, random_state=42 | Median impute (numeric); most_frequent impute + OneHotEncoder (categorical) |
| ExtraTreesClassifier | n_estimators=100, random_state=42 | Median impute (numeric); most_frequent impute + OneHotEncoder (categorical) |
| HistGradientBoostingClassifier | max_iter=100, random_state=42 | Median impute (numeric); most_frequent impute + OrdinalEncoder (categorical) |

**Source**: CONFIRMED_BY_FILE_INSPECTION of `outputs/model_compare/candidate_model_manifest.json`

### Selection Rule

- **Primary**: highest validation ROC-AUC
- **Tie-breaker 1**: validation PR-AUC
- **Tie-breaker 2**: validation F1 at threshold 0.5
- **Tie-breaker 3**: fixed priority order (LR > HGBC > RF > ET)

### Selection Result

| Rank | Model | Val ROC-AUC | Val PR-AUC | Val F1 |
|------|-------|-------------|------------|--------|
| 1 | **HistGradientBoostingClassifier** | **0.994632** | **0.993664** | **0.956265** |
| 2 | RandomForestClassifier | 0.993699 | 0.992709 | 0.955672 |
| 3 | ExtraTreesClassifier | 0.993141 | 0.992034 | 0.952718 |
| 4 | LogisticRegression | 0.926597 | 0.930832 | 0.851461 |

**Selected model**: HistGradientBoostingClassifier (won outright on validation ROC-AUC; no tie-breaking needed)

**Source**: CONFIRMED_BY_FILE_INSPECTION of `outputs/model_compare/candidate_model_manifest.json`

## Selected Model Details

- **Algorithm**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Parameters**: `max_iter=100, random_state=42` (all other parameters at sklearn defaults)
- **Preprocessing**: Median imputation for numeric features; most_frequent imputation + OrdinalEncoder for categorical features
- **Classification threshold**: fixed at 0.5
- **Total input features**: 22 (18 numeric + 4 categorical)
- **Total transformed features**: 22 (OrdinalEncoder preserves dimensionality)
- **Saved model**: `outputs/model/model.joblib`

## Evaluation Metrics

### Validation Set (used for model selection)

| Metric | Value | Source |
|--------|-------|--------|
| ROC-AUC | 0.994632 | CONFIRMED_BY_FILE_INSPECTION (`outputs/model/metrics_validation.json`) |
| PR-AUC | 0.993664 | CONFIRMED_BY_FILE_INSPECTION |
| Accuracy | 0.962735 | CONFIRMED_BY_FILE_INSPECTION |
| Precision | 0.975541 | CONFIRMED_BY_FILE_INSPECTION |
| Recall | 0.937736 | CONFIRMED_BY_FILE_INSPECTION |
| F1 | 0.956265 | CONFIRMED_BY_FILE_INSPECTION |

**Confusion matrix** (from `outputs/model/confusion_matrix_validation.csv`):

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | 10,819 | 199 |
| Actual 1 | 527 | 7,937 |

### Test Set (post-selection descriptive — NOT used for model selection)

| Metric | Value | Source |
|--------|-------|--------|
| ROC-AUC | 0.994966 | CONFIRMED_BY_FILE_INSPECTION (`outputs/model/metrics_test.json`) |
| PR-AUC | 0.994084 | CONFIRMED_BY_FILE_INSPECTION |
| Accuracy | 0.964480 | CONFIRMED_BY_FILE_INSPECTION |
| Precision | 0.975061 | CONFIRMED_BY_FILE_INSPECTION |
| Recall | 0.942344 | CONFIRMED_BY_FILE_INSPECTION |
| F1 | 0.958423 | CONFIRMED_BY_FILE_INSPECTION |

**Confusion matrix** (from `outputs/model/confusion_matrix_test.csv`):

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | 10,814 | 204 |
| Actual 1 | 488 | 7,976 |

### Metric Interpretation

- Validation-to-test gap is negligible (< 0.002 ROC-AUC), suggesting no overfitting
- Precision (0.975) is slightly higher than recall (0.942), meaning the model is more conservative about predicting satisfaction
- The model misclassifies ~5.3% of satisfied passengers as dissatisfied (527 false negatives on validation)

## Intended Use

- Benchmark comparison across LLM-assisted predictive analytics workflows
- Educational demonstration of a controlled model selection pipeline
- Baseline for further experimentation on this dataset

## Out-of-Scope Use

- Deployment for real airline customer decisions without further validation
- Causal inference about what drives passenger satisfaction
- Generalisation to other airlines, time periods, or survey instruments
- Individual-level decision-making without fairness assessment

## Limitations

1. **No hyperparameter tuning**: the model uses fixed parameters — performance could improve with tuning
2. **Near-duplicate features retained**: `Arrival Delay in Minutes` and `Departure Delay in Minutes` (r=0.966) are both included; no feature selection was performed
3. **Survey data limitations**: ratings are self-reported and may reflect survey fatigue or response bias
4. **Zero-inflation**: some 0 values in service ratings may mean "not applicable" rather than a true zero rating
5. **Single seed**: reproducibility is verified at seed=42 only
6. **No fairness assessment**: no analysis of disparate impact across demographic groups (Gender, Age)

## Risks

- **Overconfidence risk**: ROC-AUC > 0.99 may partly reflect data characteristics (e.g., service ratings directly encoding satisfaction) rather than genuine predictive difficulty
- **Distribution shift**: the model may degrade on passengers, routes, or airlines not represented in this dataset
- **Delayed feedback**: in deployment, satisfaction labels would only be available after surveys — the model cannot predict satisfaction in real time without the survey features

## Deployment Readiness

**Not production-ready.** This model is a benchmark artefact. Deployment would require:
- Fairness and bias assessment
- Robustness testing across seeds and data perturbations
- Monitoring infrastructure for distribution shift
- Validation on held-out temporal data
- Legal and ethical review of automated satisfaction predictions

## Monitoring Considerations

If deployed, monitor for:
- Class distribution drift in incoming data
- Feature distribution drift (especially delay columns which are heavily skewed)
- Prediction confidence calibration over time
- Disparate error rates across demographic subgroups

## Provenance

- **Generated by**: Claude Opus 4.6 (1M context) — benchmark workflow
- **Generation date**: 2026-03-16
- **Runtime**: UNKNOWN
- **Token usage**: UNKNOWN
- **Monetary cost**: UNKNOWN
