# EDA Summary

## Scope
- Status: CONFIRMED BY EXECUTION
- Analysis data: derived training split only
- Split rule: combined original `train.csv` and `test.csv`, then stratified 70/15/15 with seed `42`
- Training rows analysed: 90,916

## Key insights
- The training split is moderately imbalanced rather than extreme: 56.55% negative vs 43.45% positive.
- The strongest target associations are concentrated in service-quality and travel-context variables, especially Class, Online boarding, Type of Travel, Inflight entertainment, Seat comfort.
- `Type of Travel`, `Class`, and `Customer Type` show large differences in satisfaction rates across levels, so they look especially relevant for baseline modelling.
- `Flight Distance` is positively associated with satisfaction, while delay features are much weaker direct signals than onboard experience ratings.
- Several service rating features move together, which suggests useful signal but also possible redundancy in linear models.

## Data issues and modelling risks
- Missingness is concentrated in `Arrival Delay in Minutes` only: 293 rows (0.32%) in the training split.
- `id` is unique per row and `Unnamed: 0` behaves like an index column, so both should be treated as identifier-style fields.
- Delay variables are heavily right-skewed, so scale-sensitive models may benefit from robust handling or transformations.
- The strongest numeric redundancy is between `Departure Delay in Minutes` and `Arrival Delay in Minutes` (correlation 0.966).
- No duplicate rows were found in the analysed training data, and no obvious invalid rating values were detected outside the expected 0-5 range.

## Files
- Plot artefacts: `outputs/eda/`
- Summary tables: `outputs/eda/numeric_summary.csv`, `outputs/eda/categorical_summary.csv`
- Data quality log: `outputs/eda/data_quality_report.json`
