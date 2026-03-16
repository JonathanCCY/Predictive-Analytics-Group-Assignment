# Model Card: Airline Passenger Satisfaction Predictor

## Model Purpose
To predict whether an airline passenger will be 'satisfied' with their experience.

## Target Variable and Label Encoding
Target: `satisfaction`
- `satisfied` = 1
- `neutral or dissatisfied` = 0

## Split Policy and Actual Split
- Canonical split policy: stratified `70/15/15` ratio.
- Split metadata is documented in `outputs/shared/split_manifest.json`.
  - Train: 90,916 rows
  - Validation: 19,482 rows
  - Test: 19,482 rows

## Test Set Restrictions
The test set was isolated initially and played no role in feature engineering, model selection, or tuning. It was evaluated strictly once post-selection.

## Candidate Model Set and Model Selection Rule
Four candidate baseline models were evaluated:
- LogisticRegression
- RandomForestClassifier
- ExtraTreesClassifier
- HistGradientBoostingClassifier

Selection rule based exclusively on the validation set, broken by priorities in order:
1. ROC-AUC
2. PR-AUC
3. F1-score at 0.5 threshold

## Selected Model and Preprocessing
- **Selected Model:** HistGradientBoostingClassifier
- **Preprocessing:** Per pipeline definitions, missing values and categoricals were encoded via preprocessors (like OrdinalEncoder). Exact mapping strictly fitted on the train split prior to prediction to prevent information leakage.

## Evaluation Metrics (Observed)
Validation Set Metrics (HistGradientBoostingClassifier):
- ROC-AUC: 0.9948
- PR-AUC: 0.9939
- Accuracy: 0.9625
- Precision: 0.9755
- Recall: 0.9371
- F1-score: 0.9560

Test Set Metrics (HistGradientBoostingClassifier):
- ROC-AUC: 0.9951
- PR-AUC: 0.9942
- Accuracy: 0.9652
- Precision: 0.9766
- Recall: 0.9426
- F1-score: 0.9593

Unverified Metrics:
- Formal performance bounds or segmented demographic bias metrics: UNKNOWN.

## Intended Use and Out-of-Scope Use
- **Intended Use:** Baseline prediction of passenger satisfaction for service improvement insights.
- **Out-of-Scope Use:** Direct real-time operational service allocations or punitive employee evaluations.

## Limitations, Risks, and Monitoring
- **Limitations:** Features like 'Arrival Delay in Minutes' and 'Departure Delay in Minutes' are collinear.
- **Risks:** Concept drift depending on airline policy changes or general travel shifts.
- **Monitoring:** Drift detection on the distribution of service ratings is necessary.

## Deployment Readiness
Not currently described as production-ready. The model serves as a local baseline and requires external validation, fairness checks, and extensive integration testing before any production deployment.
