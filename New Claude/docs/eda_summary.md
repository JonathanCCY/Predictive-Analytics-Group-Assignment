# EDA Summary — Airline Passenger Satisfaction

## Dataset Overview

- **Source**: `data/train.csv` (103,904 rows) + `data/test.csv` (25,976 rows) combined into 129,880 rows
- **Columns dropped**: `Unnamed: 0` (Kaggle row index), `id` (row identifier)
- **Final feature set**: 22 features (18 numeric, 4 categorical) + 1 binary target
- **Target**: `satisfaction` — encoded as `satisfied` = 1, `neutral or dissatisfied` = 0
- **All EDA below uses training rows only** (90,916 rows)

## Canonical Split

| Split      | Rows   | Class 0 (%) | Class 1 (%) |
|------------|--------|-------------|-------------|
| Train      | 90,916 | 56.55%      | 43.45%      |
| Validation | 19,482 | 56.55%      | 43.45%      |
| Test       | 19,482 | 56.55%      | 43.45%      |

Stratification is exact across all three splits. Seed = 42.

## Class Balance

The target is moderately imbalanced: **56.6% neutral/dissatisfied vs 43.4% satisfied**. This is not severe enough to require resampling by default, but class-aware metrics (ROC-AUC, PR-AUC, F1) should be preferred over raw accuracy.

## Missing Values

Only one column has missing values:
- **Arrival Delay in Minutes**: 293 missing (0.32% of training set)

All other columns are complete. The missingness rate is low enough that median imputation is a reasonable default.

## Data Quality

- **Duplicates**: 0 duplicate rows in training set
- **Invalid values**: None detected (all ratings within [0, 5], no negative delays)
- **Severe skew**: `Departure Delay in Minutes` (skew = 6.85) and `Arrival Delay in Minutes` (skew = 6.71) — both heavily right-skewed with long tails (max ~1,590 minutes)
- **Near-duplicate features**: `Arrival Delay in Minutes` correlates at r = 0.966 with `Departure Delay in Minutes` — these carry nearly identical information

## Top Predictive Features (by univariate association with target)

### Strongest numeric associations (point-biserial |r|):
1. **Online boarding** — |r| = 0.464
2. **Inflight entertainment** — |r| = 0.393
3. **Seat comfort** — |r| = 0.346
4. **On-board service** — |r| = 0.319
5. **Leg room service** — |r| = 0.315
6. **Cleanliness** — |r| = 0.313

### Strongest categorical associations (Cramer's V):
1. **Type of Travel** — V = 0.458 (business travellers have 58.3% satisfaction vs 10.2% for personal)
2. **Class** — V = 0.441 (business class: 69.5% satisfied; eco: 18.8%)
3. **Customer Type** — V = 0.179 (loyal: 47.8% vs disloyal: 24.0%)
4. **Gender** — V = 0.011 (negligible difference)

### Weakest features:
- **Gate location** — |r| = 0.009 (essentially unrelated to satisfaction)
- **Gender** — V = 0.011
- **Departure/Arrival time convenient** — |r| = 0.018
- **Departure Delay in Minutes** — |r| = 0.049
- **Arrival Delay in Minutes** — |r| = 0.044

## Key Patterns

1. **Service quality ratings dominate**: The 14 service rating features (scale 0-5) are the primary discriminators. Online boarding, inflight entertainment, and seat comfort are most informative.
2. **Travel context matters**: Type of Travel and Class are strong categorical predictors. Business travellers flying in Business class are far more likely to be satisfied.
3. **Delays have weak predictive power**: Despite intuition, delay minutes have very low correlation with satisfaction (|r| < 0.05).
4. **Age has moderate signal**: Bimodal relationship — likely interacts with travel type.
5. **Flight Distance is moderately useful**: |r| = 0.167, likely correlated with class and travel type.

## Modelling Risks

1. **Redundancy**: Arrival Delay ≈ Departure Delay (r = 0.966). Consider dropping one or combining.
2. **Skewed delays**: Extreme right skew may affect linear models; tree-based models are naturally robust.
3. **Zero-inflation in ratings**: Some rating columns have a notable proportion of 0 values (which may represent "not applicable" rather than a true zero rating).
4. **Multicollinearity among service ratings**: Several service features are moderately correlated (r = 0.4-0.6), which may affect logistic regression coefficients but not tree-based models.
5. **Gender is nearly uninformative**: Can be retained but adds negligible signal.

## Artefacts Produced

All saved to `outputs/eda/`:
- `class_balance.png`, `missing_values.png`, `numeric_distributions.png`
- `correlation_heatmap.png`, `top_target_associations.png`, `categorical_vs_target.png`
- `numeric_summary.csv`, `categorical_summary.csv`
- `data_quality_report.json`, `run_log.txt`, `run_metadata.json`

Split manifest: `outputs/shared/split_manifest.json`
