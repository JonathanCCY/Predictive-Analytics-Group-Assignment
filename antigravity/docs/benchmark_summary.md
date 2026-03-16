# Benchmark Summary

## Task Artefacts Status
- **EDA**: Present (`outputs/eda/`, `docs/eda_summary.md`)
- **Model Comparison**: Present (`outputs/model_compare/`, `src/train_and_compare_models.py`)
- **Selected Model Evaluation**: Present (`outputs/model_compare/model_selection_report.md`, `outputs/model/metrics_test.json`)
- **Debugging**: Present (`src/broken_pipeline_fixed.py`, `outputs/debug/`)
- **Documentation**: Present (`README.md`, `docs/model_card.md`, `docs/decision_log.md`, `docs/reproducibility_record.md`, `docs/benchmark_summary.md`)

## Claims Verification Status
- Dataset combination and 70/15/15 stratified split: **CONFIRMED_BY_FILE_INSPECTION**
- Preprocessing fitted on training data only: **CONFIRMED_BY_EXECUTION** (based on debugging patch and log review)
- Candidate model valid selection rule application: **CONFIRMED_BY_FILE_INSPECTION**
- Debugging fixes for shape mismatch and test-set leaks: **CONFIRMED_BY_EXECUTION**
- Metrics generated for all four baseline models without test-set influence: **CONFIRMED_BY_FILE_INSPECTION**

## Unverified Claims / Missing Evidence
- Hardware execution timings / actual system loads during test run: UNVERIFIED
- Monetary costs and AI tokens logged natively across benchmark pipeline execution: UNVERIFIED

## Placeholders
- Runtime: UNKNOWN
- Token usage: UNKNOWN
- Monetary cost: UNKNOWN
