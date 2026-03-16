# Airline Passenger Satisfaction — Predictive Analytics Pipeline

## Objective

Build a reproducible binary classification pipeline to predict whether an airline passenger is **satisfied** or **neutral/dissatisfied** after a flight, using survey and flight data. The project compares four candidate models under controlled conditions and selects the best performer using a validation-governed protocol.

## Dataset

| File | Rows | Columns |
|------|------|---------|
| `data/train.csv` | 103,904 | 25 |
| `data/test.csv` | 25,976 | 25 |
| **Combined** | **129,880** | **23 (after drops)** |

**Target variable**: `satisfaction`
- `satisfied` = 1 (positive class)
- `neutral or dissatisfied` = 0 (negative class)

**Dropped columns** (before any modelling):
- `Unnamed: 0` — Kaggle row index (not a feature)
- `id` — row identifier (not a feature)

**Feature summary (22 features)**:
- 18 numeric: `Age`, `Flight Distance`, 14 service rating columns (scale 0–5), `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- 4 categorical: `Gender`, `Customer Type`, `Type of Travel`, `Class`

**Data quality notes** (from `outputs/eda/data_quality_report.json`):
- Missing values: only `Arrival Delay in Minutes` (293 rows, 0.32% of training set)
- Duplicates: 0
- Severe skew: `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- Near-duplicate pair: Arrival Delay and Departure Delay (r = 0.966)

## Folder Structure

```
.
├── data/                          # Source CSV files (do not modify)
│   ├── train.csv
│   └── test.csv
├── src/                           # All Python scripts
│   ├── eda.py                     # Step 2: EDA and split creation
│   ├── train_and_compare_models.py # Step 3: Model comparison
│   └── broken_pipeline_fixed.py   # Step 4: Corrected broken pipeline
├── outputs/
│   ├── shared/
│   │   └── split_manifest.json    # Canonical split definition (created once in Step 2)
│   ├── eda/                       # EDA plots, summaries, reports
│   ├── model_compare/             # Multi-model comparison outputs
│   ├── model/                     # Final selected model outputs
│   └── debug/                     # Corrected broken pipeline outputs
├── docs/
│   ├── eda_summary.md             # EDA insights
│   ├── model_card.md              # Model card for the selected model
│   ├── benchmark_summary.md       # Cross-task benchmark summary
│   ├── decision_log.md            # All design decisions with rationale
│   └── reproducibility_record.md  # Audit trail for all steps
├── broken_pipeline.py             # Pre-provided deliberately broken pipeline (Step 4 input)
├── requirements.txt               # Pinned Python dependencies
└── README.md
```

## Environment Setup

**Python version**: 3.11.x recommended

**Dependencies** (from `requirements.txt`):

```
numpy==1.26.4
pandas==2.2.2
scipy==1.11.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
pytest==8.2.2
```

**Install**:
```bash
pip install -r requirements.txt
```

## Canonical Split Policy

The dataset is split into **train / validation / test** at a **70 / 15 / 15** ratio using stratified sampling on the target variable.

- **Created once** in Step 2 (`src/eda.py`) and saved to `outputs/shared/split_manifest.json`
- **Reused by all subsequent steps** — never regenerated or modified
- Method: two-stage `sklearn.model_selection.train_test_split` (70/30, then 50/50 on the remainder)
- `random_state=42` at both stages
- Stratification preserves class proportions across all splits

| Split | Rows | Class 0 | Class 1 | Satisfied rate |
|-------|------|---------|---------|----------------|
| Train | 90,916 | 51,416 | 39,500 | 43.45% |
| Validation | 19,482 | 11,018 | 8,464 | 43.45% |
| Test | 19,482 | 11,018 | 8,464 | 43.45% |

## Execution Order

Run all scripts from the project root directory.

### Step 2 — EDA and Split Creation
```bash
python src/eda.py
```
- Combines data, drops identifier columns, encodes target
- Creates canonical split and saves `outputs/shared/split_manifest.json`
- Produces EDA plots and summaries in `outputs/eda/`
- All target-aware EDA uses **training rows only**

### Step 3 — Model Comparison
```bash
python src/train_and_compare_models.py
```
- Reuses split from `outputs/shared/split_manifest.json`
- Trains 4 candidate models (LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting)
- Selects final model by validation ROC-AUC
- Saves comparison outputs to `outputs/model_compare/` and final model to `outputs/model/`

### Step 4 — Debug Broken Pipeline
```bash
python src/broken_pipeline_fixed.py
```
- Corrected version of `broken_pipeline.py` (15 bugs fixed)
- Produces a LogisticRegression baseline using the canonical split
- Saves outputs to `outputs/debug/`

## Reproducibility Notes

- **Random seed**: `42` used for all sources of randomness (splits, models, numpy global seed)
- **Split integrity**: canonical split created once, stored as indices in JSON, verified by re-run stability checks
- **Leakage prevention**: all preprocessors fit on training data only; validation and test data are transformed only
- **Test-set discipline**: test metrics are reported as "post-selection descriptive" — never used for model selection, threshold tuning, or feature decisions
- **Dependency boundary**: only libraries in `requirements.txt` are used; all models are sklearn-native
- **Stability verified**: all scripts produce identical outputs on re-run (SHA-256 hash checks passed)

## Known Limitations

1. **Near-duplicate features**: `Arrival Delay in Minutes` and `Departure Delay in Minutes` correlate at r = 0.966 and are both retained (no feature selection applied)
2. **No hyperparameter tuning**: all models use fixed parameters as specified by the benchmark — no grid search or cross-validation
3. **No threshold tuning**: classification threshold fixed at 0.5
4. **Survey bias**: the dataset contains self-reported satisfaction ratings, which may reflect survey design rather than true causal factors
5. **Zero-inflation in ratings**: some service features include 0 values that may represent "not applicable" rather than the lowest rating
6. **Single seed**: reproducibility is verified at seed 42 only; robustness across seeds is not tested
7. **Not production-ready**: the model is a benchmark artefact, not validated for deployment
