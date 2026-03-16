# Airline Passenger Satisfaction Predictive Analytics

## Project Objective
This project builds a predictive analytics pipeline for a binary classification task to predict airline passenger `satisfaction` (1 = satisfied, 0 = neutral or dissatisfied).

## Dataset Inputs
- `data/train.csv` (103,904 rows originally)
- `data/test.csv` (25,976 rows originally)
- Combined size before resplitting: 129,880 rows
- Total features used for modelling: 22 (excluding `Unnamed: 0` and `id`)

## Folder Structure
- `data/`: Contains raw CSV inputs.
- `docs/`: Contains project documentation (decision log, reproducibility record, EDA summary, model card, benchmark summary).
- `src/`: Python source code scripts for EDA, modeling, and the fixed pipeline.
- `outputs/`: Saved outputs split by task:
  - `outputs/shared/`: split manifests.
  - `outputs/eda/`: EDA outputs and figures.
  - `outputs/model_compare/`: candidate model comparison outputs.
  - `outputs/model/`: final selected model and metrics.
  - `outputs/debug/`: outputs from the corrected debug pipeline.
- `tests/`: Project tests directory.

## Environment Setup and Dependencies
Dependencies are listed in `requirements.txt`.
To recreate the environment (requires Python 3.11.x):
```bash
pip install -r requirements.txt
```

## Canonical Split Policy
The canonical split is stratified on the target variable with a `70/15/15` ratio across train/validation/test splits. The exact row allocation is stored in `outputs/shared/split_manifest.json`.

## How to Run
Run the scripts from the project root in the following execution order:
1. **EDA**: `python src/eda.py`
2. **Model Comparison**: `python src/train_and_compare_models.py`
3. **Debugging Review**: `python src/broken_pipeline_fixed.py`

## Output Locations
- EDA figures and summaries: `outputs/eda/`
- Model selection reports: `outputs/model_compare/`
- Selected model joblib and evaluation metrics: `outputs/model/`
- Debugging fixes evaluation: `outputs/debug/`

## Reproducibility Notes
- **Seed**: Hardcoded to `42` globally where applicable.
- **Split**: Train/Val/Test strictly enforced via `split_manifest.json`.
- **Leakage Prevention**: All preprocessing fitted on the training split only. Specific fixes in debugging removed identified source leakage.
- **Test-set Discipline**: All model selection and tuning decisions were strictly based on the validation set.

## Known Limitations
- Runtimes, monetary costs, and token usages are UNKNOWN due to lack of local timing or API evidence.
- The `broken_pipeline.py` script was originally provided with severe defects (test-set leak, target map mismatch) which were patched natively in `broken_pipeline_fixed.py`.
