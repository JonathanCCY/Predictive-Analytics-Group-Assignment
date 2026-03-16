# Decision Log

This document records the choices made throughout the benchmark steps to maintain transparency, reproducibility, and alignment with the benchmark requirements.

## Step 1: Initialization
- Defined the project directory structure as specified in the benchmark instructions.
- Moved and renamed source dataset files (`train (1).csv` and `test (1).csv`) into the `data/` folder as `train.csv` and `test.csv`.
- Found `requirements.txt`, `broken_pipeline.py`, and `reproducibility_record.md`.
- `agent_reproducibility_protocol_v3.md` was **not found** in the workspace. Falling back to the rules in the main prompt as the binding specification.
- Evaluated pre-run state and recorded observations in `docs/reproducibility_record.md`.


## Step 2: EDA and Insight Generation
- Mapped satisfaction to 1 (satisfied) and 0 (neutral or dissatisfied).
- Dropped first unnamed column and id.
- Verified class balance on training dataset (70% stratified split).
- Extracted top associations using Point-Biserial correlation (numeric) and Cramer's V (categorical).


## Step 3: Controlled Baseline and Multi-Model Comparison
- Checked for presence of  before instantiating models.
- Ensured fixed threshold of  across all metric calculations.
- Maintained candidate parameters exactly to specification.
- Tested on Val and Test dataset strictly once for comparison purposes.
- Selected final model automatically using pre-coded priority logic focusing on Val ROC-AUC -> PR-AUC -> F1-score -> Fixed string logic.

## Step 3: Controlled Baseline and Multi-Model Comparison
- Checked for presence of `outputs/shared/split_manifest.json` before instantiating models.
- Ensured fixed threshold of 0.5 across all metric calculations.
- Maintained candidate parameters exactly to specification.
- Tested on Val and Test dataset strictly once for comparison purposes.
- Selected final model automatically using pre-coded priority logic focusing on Val ROC-AUC -> PR-AUC -> F1-score -> Fixed string logic.

## Step 4: Debugging Deliberately Broken Pipeline
- Identified target encoding mapping string bugs (`lambda x: 1 if x == 'satisfied' else 0` produced integers but was mixed with string logic).
- Identified methodological split leak (`test_size=0.3` without seed, overriding canonical split mapping).
- Identified methodological preprocessing leak (`preprocessor.fit_transform(X)` before split).
- Identified coding bug in training mapping (`model.fit(X_train, y_temp)` mismatched dimensions).
- Identified reproducibility error in saving model via Pickle instead of `joblib`.
- Patched all defects natively and regenerated pipeline evaluation directly to `outputs/debug/`.

## Step 5: Documentation
- Verified existence of EDA, multi-model outputs, and debug metrics strictly using local files.
- Replaced missing or unobserved data with `UNKNOWN` to remain conservative regarding costs and timings.
- Prepared `README.md`, `docs/model_card.md`, and `docs/benchmark_summary.md` exactly as requested by prompt.
- Answered prompt output format to provide the final summary.
