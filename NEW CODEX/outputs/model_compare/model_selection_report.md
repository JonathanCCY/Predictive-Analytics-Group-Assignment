# Model Selection Report

## Selection rule
- Primary metric: validation ROC-AUC
- Tie-breaker 1: validation PR-AUC
- Tie-breaker 2: validation F1 at threshold 0.5
- Tie-breaker 3: fixed priority order `LogisticRegression > HistGradientBoostingClassifier > RandomForestClassifier > ExtraTreesClassifier`
- Classification threshold fixed at `0.5` for all reported precision/recall/F1 metrics

## Selected model
- Selected model: `HistGradientBoostingClassifier`
- Validation ROC-AUC: `0.994632`
- Validation PR-AUC: `0.993664`
- Validation F1: `0.956265`

## Validation ranking
```text
                    model_name  roc_auc   pr_auc  accuracy  precision   recall       f1  threshold    tn   fp   fn   tp selection_metric
HistGradientBoostingClassifier 0.994632 0.993664  0.962735   0.975541 0.937736 0.956265        0.5 10819  199  527 7937       validation
        RandomForestClassifier 0.993699 0.992709  0.962068   0.970635 0.941163 0.955672        0.5 10777  241  498 7966       validation
          ExtraTreesClassifier 0.993141 0.992034  0.959552   0.967935 0.937973 0.952718        0.5 10755  263  525 7939       validation
            LogisticRegression 0.926597 0.930832  0.873473   0.868897 0.834712 0.851461        0.5  9952 1066 1399 7065       validation
```

## Test metrics note
- The test metrics table is post-selection descriptive comparison only and was not used for model selection.

## Test metrics by candidate model
```text
                    model_name  roc_auc   pr_auc  accuracy  precision   recall       f1  threshold    tn   fp   fn   tp                            evaluation_note
HistGradientBoostingClassifier 0.994966 0.994084  0.964480   0.975061 0.942344 0.958423        0.5 10814  204  488 7976 post-selection descriptive comparison only
        RandomForestClassifier 0.994053 0.992990  0.963299   0.971637 0.943053 0.957132        0.5 10785  233  482 7982 post-selection descriptive comparison only
          ExtraTreesClassifier 0.993102 0.991953  0.961041   0.969531 0.939863 0.954466        0.5 10768  250  509 7955 post-selection descriptive comparison only
            LogisticRegression 0.927669 0.931665  0.873986   0.871155 0.833176 0.851742        0.5  9975 1043 1412 7052 post-selection descriptive comparison only
```
