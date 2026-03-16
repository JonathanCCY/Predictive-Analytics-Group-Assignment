# Model Selection Report

## Candidate Models

| Model | Preprocessing |
|-------|--------------|
| LogisticRegression | median impute + StandardScaler for numeric; most_frequent impute + OneHotEncoder for categorical |
| RandomForestClassifier | median impute for numeric; most_frequent impute + OneHotEncoder for categorical |
| ExtraTreesClassifier | median impute for numeric; most_frequent impute + OneHotEncoder for categorical |
| HistGradientBoostingClassifier | median impute for numeric; most_frequent impute + OrdinalEncoder for categorical |

## Validation Metrics

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|--------|----------|-----------|--------|-----|
| LogisticRegression | 0.926597 | 0.930832 | 0.873473 | 0.868897 | 0.834712 | 0.851461 |
| RandomForestClassifier | 0.993699 | 0.992709 | 0.962068 | 0.970635 | 0.941163 | 0.955672 |
| ExtraTreesClassifier | 0.993141 | 0.992034 | 0.959552 | 0.967935 | 0.937973 | 0.952718 |
| HistGradientBoostingClassifier | 0.994632 | 0.993664 | 0.962735 | 0.975541 | 0.937736 | 0.956265 |

## Selection Rule

- Primary: validation ROC-AUC (higher is better)
- Tie-breaker 1: validation PR-AUC
- Tie-breaker 2: validation F1 at threshold 0.5
- Tie-breaker 3: fixed priority order (LR > HGBC > RF > ET)

## Selection Ranking

1. HistGradientBoostingClassifier — ROC-AUC=0.994632, PR-AUC=0.993664, F1=0.956265 **SELECTED**
2. RandomForestClassifier — ROC-AUC=0.993699, PR-AUC=0.992709, F1=0.955672
3. ExtraTreesClassifier — ROC-AUC=0.993141, PR-AUC=0.992034, F1=0.952718
4. LogisticRegression — ROC-AUC=0.926597, PR-AUC=0.930832, F1=0.851461

**Final selected model: HistGradientBoostingClassifier**

## Test Metrics (Post-Selection Descriptive Comparison)

These test metrics are reported **after** model selection and were **not** used for selection.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|--------|----------|-----------|--------|-----|
| LogisticRegression | 0.927669 | 0.931665 | 0.873986 | 0.871155 | 0.833176 | 0.851742 |
| RandomForestClassifier | 0.994053 | 0.992990 | 0.963299 | 0.971637 | 0.943053 | 0.957132 |
| ExtraTreesClassifier | 0.993102 | 0.991953 | 0.961041 | 0.969531 | 0.939863 | 0.954466 |
| HistGradientBoostingClassifier | 0.994966 | 0.994084 | 0.964480 | 0.975061 | 0.942344 | 0.958423 |

## Threshold

Classification threshold fixed at 0.5. No threshold tuning was performed.
