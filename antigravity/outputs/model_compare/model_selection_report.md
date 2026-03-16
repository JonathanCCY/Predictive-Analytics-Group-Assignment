# Model Selection Report

## Context
Four baseline candidate models were trained strictly according to the benchmark specifications to predict passenger `satisfaction`.

## Models Evaluated
1. LogisticRegression
2. RandomForestClassifier
3. ExtraTreesClassifier
4. HistGradientBoostingClassifier

## Selection Mechanism
Models were evaluated exclusively on the validation set.
- Primary standard: **ROC-AUC**
- Tie 1: PR-AUC
- Tie 2: F1-score at 0.5 threshold

### Results (Validation)
| Model                          |   ROC-AUC |   PR-AUC |   Accuracy |   Precision |   Recall |   F1-score |
|:-------------------------------|----------:|---------:|-----------:|------------:|---------:|-----------:|
| LogisticRegression             |  0.926597 | 0.930832 |   0.873473 |    0.868897 | 0.834712 |   0.851461 |
| RandomForestClassifier         |  0.993481 | 0.992575 |   0.962376 |    0.97192  | 0.940572 |   0.955989 |
| ExtraTreesClassifier           |  0.992741 | 0.991683 |   0.958423 |    0.966821 | 0.936437 |   0.951386 |
| HistGradientBoostingClassifier |  0.994845 | 0.993889 |   0.962478 |    0.975526 | 0.937146 |   0.955951 |

## Conclusion
The model chosen for final evaluation on the test set is **HistGradientBoostingClassifier**.
All post-selection metrics on the test set for all models are located in `outputs/model_compare/test_metrics_by_model.csv` strictly as an observational comparison without exerting selection influence.
