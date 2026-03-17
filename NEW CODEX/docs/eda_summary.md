# EDA Summary

## Split and scope
- Canonical split method: `two_stage_stratified_train_validation_test_split_70_15_15`
- Random seed: `42`
- Training rows used for EDA: `90916`
- Validation rows counted only for split verification: `19482`
- Test rows counted only for split verification: `19482`

## Most likely predictive signals
- `Class` via `cramers_v` with association `0.504`
- `Online boarding` via `point_biserial` with association `0.501`
- `Type of Travel` via `cramers_v` with association `0.449`
- `Inflight entertainment` via `point_biserial` with association `0.400`
- `Seat comfort` via `point_biserial` with association `0.349`
- `On-board service` via `point_biserial` with association `0.323`

## Data quality observations
- Missingness is concentrated in `Arrival Delay in Minutes` with `293` missing training rows.
- Duplicate training rows after dropping identifier columns: `0`.
- Invalid value flags raised: `1`.
- Identifier-like columns removed before modelling: `Unnamed: 0, id`.

## Modelling risks and caveats
- Severe skew: Departure Delay in Minutes, Arrival Delay in Minutes
- High-cardinality categoricals: none
- Possible leakage columns: none after dropping identifiers

## Notes
- Target-aware summaries and plots use training rows only.
- Validation and test splits were used only to verify split sizes and class distributions.
- Arrival Delay in Minutes is the only column with missing values in the source data.
- Customer Type labels are semantically consistent but use mixed capitalization.
- Delay variables may be operationally unavailable in pre-flight deployment settings, although they are not leakage under a post-flight satisfaction prediction framing.

## Execution status
- `CONFIRMED_BY_EXECUTION`
