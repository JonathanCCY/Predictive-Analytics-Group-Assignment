# Benchmark Summary — Airline Passenger Satisfaction

## Agent Identity

- **Agent**: Claude Opus 4.6 (1M context)
- **Execution date**: 2026-03-16
- **Runtime**: UNKNOWN
- **Token usage**: UNKNOWN
- **Monetary cost**: UNKNOWN

---

## Task Completion Summary

| Step | Task | Status | Script | Mandatory Artefacts | Created |
|------|------|--------|--------|-------------------|---------|
| 2 | EDA and Insight Generation | CONFIRMED_BY_EXECUTION | `src/eda.py` | 10 | 10/10 |
| 3 | Model Comparison | CONFIRMED_BY_EXECUTION | `src/train_and_compare_models.py` | 12 | 12/12 |
| 4 | Debug Broken Pipeline | CONFIRMED_BY_EXECUTION | `src/broken_pipeline_fixed.py` | 4 | 4/4 |
| 5 | Documentation | CONFIRMED_BY_EXECUTION | N/A | 5 | 5/5 |

**Overall mandatory artefact completion: 31/31 (100%)**

---

## Artefact Inventory

### Step 2 — EDA

| Artefact | Exists | Evidence Level |
|----------|--------|---------------|
| `outputs/shared/split_manifest.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/class_balance.png` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/missing_values.png` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/numeric_summary.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/categorical_summary.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/numeric_distributions.png` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/top_target_associations.png` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/correlation_heatmap.png` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/data_quality_report.json` | Yes | CONFIRMED_BY_EXECUTION |
| `docs/eda_summary.md` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/run_log.txt` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/run_metadata.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/eda/categorical_vs_target.png` | Yes | CONFIRMED_BY_EXECUTION |

### Step 3 — Model Comparison

| Artefact | Exists | Evidence Level |
|----------|--------|---------------|
| `outputs/model_compare/validation_metrics_by_model.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model_compare/test_metrics_by_model.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model_compare/candidate_model_manifest.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model_compare/model_selection_report.md` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/validation_predictions.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/test_predictions.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/metrics_validation.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/metrics_test.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/confusion_matrix_validation.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/confusion_matrix_test.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/model.joblib` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/feature_manifest.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model_compare/run_log.txt` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model_compare/run_metadata.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/run_metadata.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/model/run_log.txt` | Yes | CONFIRMED_BY_EXECUTION |

### Step 4 — Debug Broken Pipeline

| Artefact | Exists | Evidence Level |
|----------|--------|---------------|
| `src/broken_pipeline_fixed.py` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/metrics_validation.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/metrics_test.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/run_log.txt` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/validation_predictions.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/test_predictions.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/confusion_matrix_validation.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/confusion_matrix_test.csv` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/model.joblib` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/split_manifest.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/feature_manifest.json` | Yes | CONFIRMED_BY_EXECUTION |
| `outputs/debug/run_metadata.json` | Yes | CONFIRMED_BY_EXECUTION |

### Step 5 — Documentation

| Artefact | Exists | Evidence Level |
|----------|--------|---------------|
| `README.md` | Yes | CONFIRMED_BY_EXECUTION |
| `docs/model_card.md` | Yes | CONFIRMED_BY_EXECUTION |
| `docs/benchmark_summary.md` | Yes | CONFIRMED_BY_EXECUTION |
| `docs/reproducibility_record.md` | Yes | CONFIRMED_BY_EXECUTION |
| `docs/decision_log.md` | Yes | CONFIRMED_BY_EXECUTION |

---

## Key Metrics Summary

### Selected Model: HistGradientBoostingClassifier

| Metric | Validation | Test (post-selection) | Source |
|--------|-----------|----------------------|--------|
| ROC-AUC | 0.994632 | 0.994966 | CONFIRMED_BY_FILE_INSPECTION |
| PR-AUC | 0.993664 | 0.994084 | CONFIRMED_BY_FILE_INSPECTION |
| Accuracy | 0.962735 | 0.964480 | CONFIRMED_BY_FILE_INSPECTION |
| Precision | 0.975541 | 0.975061 | CONFIRMED_BY_FILE_INSPECTION |
| Recall | 0.937736 | 0.942344 | CONFIRMED_BY_FILE_INSPECTION |
| F1 | 0.956265 | 0.958423 | CONFIRMED_BY_FILE_INSPECTION |

### All Candidates — Validation ROC-AUC

| Model | Val ROC-AUC | Test ROC-AUC (descriptive) |
|-------|-------------|---------------------------|
| HistGradientBoostingClassifier | 0.994632 | 0.994966 |
| RandomForestClassifier | 0.993699 | 0.994053 |
| ExtraTreesClassifier | 0.993141 | 0.993102 |
| LogisticRegression | 0.926597 | 0.927669 |

### Debug Pipeline (LogisticRegression baseline)

| Metric | Validation | Test | Source |
|--------|-----------|------|--------|
| ROC-AUC | 0.926597 | 0.927669 | CONFIRMED_BY_FILE_INSPECTION |
| F1 | 0.851461 | 0.851742 | CONFIRMED_BY_FILE_INSPECTION |

Debug LR metrics match Step 3 LR exactly: CONFIRMED_BY_EXECUTION

---

## Reproducibility Checks

| Check | Result |
|-------|--------|
| Canonical split created in Step 2 only | CONFIRMED_BY_EXECUTION |
| Split reused without modification in Steps 3–4 | CONFIRMED_BY_EXECUTION |
| Consistent target encoding across all tasks | CONFIRMED_BY_EXECUTION |
| Test-set discipline preserved (no selection/tuning on test) | CONFIRMED_BY_EXECUTION |
| Preprocessors fit on training data only | CONFIRMED_BY_EXECUTION |
| Dependency boundary respected (requirements.txt) | CONFIRMED_BY_FILE_INSPECTION |
| All code-producing tasks executed (not just drafted) | CONFIRMED_BY_EXECUTION |
| Re-run stability (Step 2) | PASSED — identical split indices |
| Re-run stability (Step 3) | PASSED — identical metric file hashes |
| Re-run stability (Step 4) | PASSED — identical metrics on re-run |
| Cross-task consistency (Step 4 LR = Step 3 LR) | PASSED — metrics match exactly |

---

## Bugs Found in Broken Pipeline (Step 4)

| # | Bug | Classification |
|---|-----|---------------|
| B1 | Identifier columns not dropped | Methodological flaw |
| B2 | Preprocessor fit on entire dataset (data leakage) | Methodological flaw |
| B3 | No random_state on train_test_split | Reproducibility defect |
| B4 | Canonical split not reused | Reproducibility defect |
| B5 | Split after preprocessing (ordering error) | Methodological flaw |
| B6 | max_iter=500 instead of 1000 | Reproducibility defect |
| B7 | No random_state on LogisticRegression | Reproducibility defect |
| B8 | No explicit solver='lbfgs' | Reproducibility defect |
| B9 | model.fit(X_train, y_temp) — shape mismatch crash | Coding bug (FATAL) |
| B10 | Validation report mislabelled as "Test Set Performance" | Coding bug + Methodological flaw |
| B11 | Test set never evaluated | Methodological flaw |
| B12 | pickle.dump with text mode 'w' — crash | Coding bug (FATAL) |
| B13 | pickle instead of joblib | Reproducibility defect |
| B14 | Wrong output path | Coding bug |
| B15 | No metrics saved to files | Methodological flaw |

**Total: 15 bugs (2 fatal crashes, 5 methodological flaws, 6 reproducibility defects, 4 coding bugs)**

---

## Missing or Unverifiable Evidence

| Item | Status |
|------|--------|
| `agent_reproducibility_protocol_v3.md` | NOT FOUND — prompt rules used as fallback |
| Runtime per step | UNKNOWN |
| Total runtime | UNKNOWN |
| Token usage | UNKNOWN |
| Monetary cost | UNKNOWN |
| Robustness across seeds (seed != 42) | NOT TESTED |
| Fairness / disparate impact assessment | NOT PERFORMED |
| Temporal validation | NOT PERFORMED |

---

## Items for Human Evaluator to Check

1. **Split manifest integrity**: verify `outputs/shared/split_manifest.json` row counts match expectations (90,916 / 19,482 / 19,482)
2. **Metric file contents**: spot-check `outputs/model/metrics_validation.json` and `outputs/model/metrics_test.json` values
3. **Test-set discipline**: confirm test metrics are labelled "post-selection descriptive" in all output files
4. **Cross-step consistency**: confirm debug LR metrics match Step 3 LR metrics
5. **Preprocessing correctness**: verify HGBC uses OrdinalEncoder (not OneHotEncoder) for categoricals
6. **Bug count and classification**: review the 15-bug inventory against `broken_pipeline.py` source
7. **Re-run reproducibility**: optionally re-run `python src/eda.py && python src/train_and_compare_models.py && python src/broken_pipeline_fixed.py` and compare output hashes
