# Model Card: Airline Passenger Satisfaction Selected Model

## Model Purpose

This model card documents the **selected final model** produced by the repository's controlled comparison workflow for predicting airline passenger `satisfaction`.

Documented status: the artefacts describe a benchmark pipeline result, not a production deployment claim.

## Target Variable

- target column: `satisfaction`
- task type: binary classification
- label encoding used by the pipeline:
  - `neutral or dissatisfied` -> `0`
  - `satisfied` -> `1`

## Data and Split Policy

### Intended split policy

The intended split policy is the canonical manifest stored at `outputs/shared/split_manifest.json`:

- combined dataset source: `data/train.csv` + `data/test.csv`
- split method: `two_stage_stratified_train_validation_test_split_70_15_15`
- random seed: `42`
- target name: `satisfaction`

### Actual split used

Saved EDA, model-comparison, and debug metadata are consistent with reuse of the canonical manifest.

Actual split counts:

- train: `90916`
- validation: `19482`
- test: `19482`

Observed manifest hash reused in later steps:

- `687510b33da154d45ae88e4425fadf5f43791b592bc8978b83a4b1b47a7b99f2`

### Test-set restriction

The repository evidence supports the following restrictions:

- test rows are not used to create the canonical split after Step 2
- test rows are not used for model selection
- test rows are not used for threshold tuning
- preprocessors are fitted on training data only
- candidate-model test metrics in `outputs/model_compare/test_metrics_by_model.csv` are labelled `post-selection descriptive comparison only`
- the selected-model test evaluation in `outputs/model/metrics_test.json` is stored as a separate formal evaluation output

## Candidate Model Set and Selection Rule

Candidate models listed in `outputs/model_compare/candidate_model_manifest.json`:

- `LogisticRegression`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `HistGradientBoostingClassifier`

Fixed selection rule from saved artefacts:

1. validation ROC-AUC
2. validation PR-AUC
3. validation F1 at threshold `0.5`
4. fixed priority order: `LogisticRegression > HistGradientBoostingClassifier > RandomForestClassifier > ExtraTreesClassifier`

## Selected Model

- selected model: `HistGradientBoostingClassifier`
- selection status: `CONFIRMED_BY_EXECUTION`

### Selected-model preprocessing

Locally verifiable evidence from `src/train_and_compare_models.py`, `outputs/model_compare/candidate_model_manifest.json`, and `outputs/model/feature_manifest.json` indicates:

- numeric preprocessing: median imputation
- categorical preprocessing: most-frequent imputation plus ordinal encoding with unknown categories mapped to `-1`
- transformed feature count saved in `outputs/model/feature_manifest.json`: `22`

Original categorical features:

- `Gender`
- `Customer Type`
- `Type of Travel`
- `Class`

Original numeric features:

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

## Metrics

### Planned metrics

The saved comparison and model evaluation artefacts show that the pipeline was designed to report:

- ROC-AUC
- PR-AUC
- accuracy
- precision
- recall
- F1
- threshold
- confusion-matrix counts: `tn`, `fp`, `fn`, `tp`

### Observed metrics from saved outputs

#### Selected model on validation split

Source: `outputs/model/metrics_validation.json`

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9946316847135519 |
| PR-AUC | 0.9936636232680085 |
| Accuracy | 0.9627348321527563 |
| Precision | 0.9755408062930186 |
| Recall | 0.9377362948960303 |
| F1 | 0.9562650602409638 |
| Threshold | 0.5 |
| TN | 10819 |
| FP | 199 |
| FN | 527 |
| TP | 7937 |

#### Selected model on test split

Source: `outputs/model/metrics_test.json`

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9949659836576065 |
| PR-AUC | 0.9940840626276425 |
| Accuracy | 0.9644800328508367 |
| Precision | 0.9750611246943766 |
| Recall | 0.94234404536862 |
| F1 | 0.958423455900024 |
| Threshold | 0.5 |
| TN | 10814 |
| FP | 204 |
| FN | 488 |
| TP | 7976 |

#### Candidate-model comparison snapshot

Observed validation ROC-AUC ranking from `outputs/model_compare/validation_metrics_by_model.csv`:

| Model | Validation ROC-AUC | Validation PR-AUC | Validation F1 |
| --- | ---: | ---: | ---: |
| HistGradientBoostingClassifier | 0.9946316847135519 | 0.9936636232680085 | 0.9562650602409638 |
| RandomForestClassifier | 0.9936987455824993 | 0.9927093579750123 | 0.9556715254033952 |
| ExtraTreesClassifier | 0.993141180345549 | 0.992033844884961 | 0.9527181087243489 |
| LogisticRegression | 0.926597096570966 | 0.9308320819336684 | 0.8514612835191323 |

### Unverified metrics

The following were not found in saved artefacts and are therefore `UNKNOWN`:

- calibration metrics
- subgroup or fairness metrics
- external-validation metrics
- post-deployment monitoring thresholds
- runtime per training run
- token usage
- monetary cost

## Intended Use

Appropriate uses supported by repository evidence:

- benchmarked comparison of four sklearn-native classifiers on the provided airline satisfaction dataset
- reproducible local reruns using the canonical split manifest
- teaching, audit, or review of validation-based model selection and saved evaluation artefacts

## Out-of-Scope Use

This repository does not provide evidence for:

- real-time production deployment
- medical, legal, or safety-critical decision support
- causal inference
- fairness-sensitive deployment without additional review
- use on materially different populations without external validation

## Limitations and Risks

- evaluation is limited to the provided dataset and canonical split
- repeated-run stability checks are documented as `NOT_EXECUTED`
- a separate standalone final-evaluation script was not found
- `Customer Type` contains mixed capitalization in the raw data
- delay-related features may not be available in every operational setting
- no subgroup performance analysis was found
- no monitoring plan with numerical alert thresholds was found
- no evidence justifies describing the model as production-ready

## Monitoring Considerations

If this benchmark model is ever considered for broader use, a human reviewer should at least check:

- schema consistency for all 22 modelling features
- label encoding consistency for `satisfaction`
- drift in class balance and delay-feature distributions
- performance by relevant passenger subgroups
- whether delay variables are available at the intended prediction time
- whether the canonical split and selection rule remain unchanged

## Deployment Readiness

Current status: **not demonstrated as production-ready**.

Reason:

- the repository provides benchmark-grade training and evaluation artefacts
- it does not provide external validation, monitoring thresholds, fairness review, service integration evidence, or deployment controls
