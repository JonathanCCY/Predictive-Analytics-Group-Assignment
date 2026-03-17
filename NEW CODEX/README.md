# Airline Passenger Satisfaction Predictive Analytics Pipeline

## Objective

This repository contains a reproducible predictive analytics workflow for the **Airline Passenger Satisfaction** dataset. The prediction task is binary classification of `satisfaction`, with project outputs covering:

- exploratory data analysis (EDA)
- controlled candidate-model comparison
- selected-model evaluation
- debugging review artefacts for a corrected baseline pipeline

The documentation in this repository is restricted to information that could be verified locally from code, saved outputs, logs, and metadata. If a detail could not be verified, it is marked as `UNKNOWN`.

## Dataset Inputs

### Source files

| Path | Repository state | Rows | Notes |
| --- | --- | ---: | --- |
| `data/train.csv` | symlink to `../train.csv` | 103,904 | confirmed by local file inspection |
| `data/test.csv` | symlink to `../test.csv` | 25,976 | confirmed by local file inspection |

Combined source size before resplitting: **129,880 rows**

### Raw columns

The raw CSV header contains 25 columns:

- unnamed first column
- `id`
- 22 modelling features
- `satisfaction` target

### Modelling column summary

Columns dropped before modelling:

- unnamed first column (`Unnamed: 0` / first unnamed index column)
- `id`

Target column:

- `satisfaction`

Target encoding:

- `neutral or dissatisfied` -> `0`
- `satisfied` -> `1`

Categorical feature columns (4):

- `Gender`
- `Customer Type`
- `Type of Travel`
- `Class`

Numeric feature columns (18):

- `Age`
- `Flight Distance`
- `Inflight wifi service`
- `Departure/Arrival time convenient`
- `Ease of Online booking`
- `Gate location`
- `Food and drink`
- `Online boarding`
- `Seat comfort`
- `Inflight entertainment`
- `On-board service`
- `Leg room service`
- `Baggage handling`
- `Checkin service`
- `Inflight service`
- `Cleanliness`
- `Departure Delay in Minutes`
- `Arrival Delay in Minutes`

Observed data quality notes from saved EDA outputs:

- only `Arrival Delay in Minutes` has missing values in the training split summary, with 293 missing rows
- duplicate training rows after dropping identifier columns: 0
- `Customer Type` uses mixed capitalization (`Loyal Customer` and `disloyal Customer`)
- severe skew was flagged for `Departure Delay in Minutes` and `Arrival Delay in Minutes`

## Folder Structure

```text
.
|-- README.md
|-- requirements.txt
|-- train.csv
|-- test.csv
|-- data/
|   |-- train.csv -> ../train.csv
|   `-- test.csv -> ../test.csv
|-- src/
|   |-- bootstrap_benchmark.py
|   |-- eda.py
|   |-- train_and_compare_models.py
|   `-- broken_pipeline_fixed.py
|-- outputs/
|   |-- shared/
|   |   `-- split_manifest.json
|   |-- eda/
|   |-- model_compare/
|   |-- model/
|   |-- debug/
|   `-- step1_setup/
`-- docs/
    |-- benchmark_summary.md
    |-- decision_log.md
    |-- eda_summary.md
    |-- model_card.md
    `-- reproducibility_record.md
```

## Environment Setup

Dependency scope is defined in `requirements.txt`.

- recommended interpreter noted in `requirements.txt`: Python `3.11.x`
- pinned libraries listed in `requirements.txt`:
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `scipy==1.11.4`
  - `scikit-learn==1.4.2`
  - `matplotlib==3.8.4`
  - `seaborn==0.13.2`
  - `joblib==1.4.2`
  - `pytest==8.2.2`

Conventional pip-based setup inferred from the presence of `requirements.txt` (not confirmed by saved run logs):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Canonical Split Policy

The canonical split manifest is stored at `outputs/shared/split_manifest.json`.

- split policy: `two_stage_stratified_train_validation_test_split_70_15_15`
- seed: `42`
- total rows covered by the manifest: `129880`
- actual split counts from saved metadata:
  - train: `90916`
  - validation: `19482`
  - test: `19482`

This manifest is created in the EDA step and reused in later steps. Saved model-comparison and debug metadata both record reuse of the same manifest hash:

- `687510b33da154d45ae88e4425fadf5f43791b592bc8978b83a4b1b47a7b99f2`

## Execution Order

Run the workflow in this order:

1. EDA and canonical split creation
2. Candidate-model comparison
3. Final model evaluation
4. Debugging review of the deliberately broken pipeline

Repository-backed commands:

```bash
python3 src/eda.py
python3 src/train_and_compare_models.py
python3 src/broken_pipeline_fixed.py
```

### Step details

#### 1. EDA

Command:

```bash
python3 src/eda.py
```

This step:

- combines `data/train.csv` and `data/test.csv`
- drops the unnamed first column and `id`
- creates `outputs/shared/split_manifest.json` if it does not already exist
- performs target-aware EDA on the training split only
- writes plots, summaries, and run metadata under `outputs/eda/`

#### 2. Model comparison

Command:

```bash
python3 src/train_and_compare_models.py
```

This step:

- reuses `outputs/shared/split_manifest.json`
- trains the four candidate models listed in `outputs/model_compare/candidate_model_manifest.json`
- selects the final model using validation ROC-AUC, then validation PR-AUC, then validation F1 at threshold `0.5`, then a fixed priority order
- writes comparison artefacts under `outputs/model_compare/`
- writes selected-model artefacts under `outputs/model/`

#### 3. Final model evaluation

Standalone checked-in entrypoint: `UNKNOWN`

Observed repository behavior:

- the selected-model evaluation artefacts under `outputs/model/` are produced by `python3 src/train_and_compare_models.py`
- saved outputs include `outputs/model/metrics_validation.json`, `outputs/model/metrics_test.json`, prediction files, confusion matrices, a feature manifest, and `outputs/model/model.joblib`

#### 4. Debugging review

Command:

```bash
python3 src/broken_pipeline_fixed.py
```

This step:

- inspects the root-level `broken_pipeline.py`
- reuses the canonical split manifest
- preserves the Logistic Regression baseline intent while correcting the broken pipeline
- writes debug artefacts under `outputs/debug/`

## Output Locations

| Output area | Contents |
| --- | --- |
| `outputs/shared/` | canonical split manifest |
| `outputs/eda/` | plots, numeric summary, categorical summary, data quality report, run metadata |
| `outputs/model_compare/` | candidate model manifest, validation metrics table, descriptive test metrics table, selection report, run metadata |
| `outputs/model/` | selected-model metrics, predictions, confusion matrices, feature manifest, serialized model, run metadata |
| `outputs/debug/` | corrected Logistic Regression baseline artefacts and debug run metadata |
| `docs/` | narrative documentation and audit trail |

## Reproducibility Notes

- Fixed random seed: `42`
- Canonical split is created once in Step 2 and then reused.
- The split manifest stores row assignments for train, validation, and test.
- EDA is training-split only for target-aware analysis.
- Model selection is validation-based only.
- Reported precision, recall, and F1 use a fixed classification threshold of `0.5`.
- Saved test metrics for all candidates in `outputs/model_compare/test_metrics_by_model.csv` are explicitly labelled as post-selection descriptive comparison only.
- The formal selected-model test evaluation is stored separately in `outputs/model/metrics_test.json`.
- Preprocessing is defined in code and fitted on training data only.

## Known Limitations

- No separate checked-in script for a standalone final-model-evaluation step was found; selected-model evaluation is coupled to `src/train_and_compare_models.py`.
- Repeated-run stability checks are documented as `NOT_EXECUTED` in `docs/reproducibility_record.md`.
- Runtime, token usage, and monetary cost are `UNKNOWN`.
- No automated test files were found under `tests/`.
- Delay features may be unavailable in some pre-flight deployment settings, even though saved EDA notes do not classify them as leakage for a post-flight satisfaction framing.
- The repository contains both a root-level `reproducibility_record.md` and the active `docs/reproducibility_record.md`, which a human maintainer may want to reconcile.
