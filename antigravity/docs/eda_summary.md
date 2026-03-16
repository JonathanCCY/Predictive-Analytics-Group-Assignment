# EDA Insight Summary

This document sets out the key insights derived from the exploratory data analysis (EDA) performed exclusively on the model training split (70% of the total dataset, 90,916 rows) for the Airline Passenger Satisfaction task.

## 1. Dataset Shape and Context
- **Total Combined Dataset:** 129,880 rows.
- **EDA Split (Train Only):** 90,916 rows.
- **Features Used:** 22 predictive features after discarding `Unnamed: 0` and `id`. 

## 2. Target Variable Assessment
The `satisfaction` target variable was encoded as 1 for `satisfied` and 0 for `neutral or dissatisfied`. The current class distribution in the training set is:
- **0 (neutral or dissatisfied):** 51,416 rows (56.5%)
- **1 (satisfied):** 39,500 rows (43.5%)

Class balance is relatively healthy and does not formally demand severe imbalance techniques like SMOTE or undersampling, although slight weighting during tree-based model training could be considered.

## 3. Data Quality Issues
- **Missing Values:** Missingness is rare but isolated to **`Arrival Delay in Minutes`** (293 missing entries, ~0.3%). This column will require median or iterative imputation before model training.
- **Duplicates:** No duplicated rows were detected.
- **Invalid Values / High Cardinality:** No obvious high-cardinality categorical features (>20 distinct values) or immediately invalid outliers beyond structural skewness.
- **Severe Skewness:** Both **`Departure Delay in Minutes`** and **`Arrival Delay in Minutes`** display severe positive right-skewness (skew > 3). Transformations (e.g., Log1p or Box-Cox) may be required for linear models though robust tree-based models might handle them gracefully.

## 4. Feature Dependencies and Leakage Risks
- `Departure Delay in Minutes` and `Arrival Delay in Minutes` intuitively capture very similar occurrences. There may be collinearity risks during logistic regression, necessitating regularization or removal of one.
- No direct target leakage was formally identified among the features, however the strong behavioral proxy signals in service-oriented categorical responses might dominate baseline models.

## 5. Potential Predictive Drivers
By inspecting univariate associations on the training data:
- Features related to the direct in-flight experience (e.g., `Inflight wifi service`, `Online boarding`, `Class`, and `Type of Travel`) generally possess the highest correlative strengths (Cramér's V or Abs. Point-Biserial Correlation) with passenger satisfaction. 

### Conclusion
The dataset is largely clean and properly scaled for robust model training. Priority should be given to handling the minimal null values in the target-related delay column and being mindful of the collinearity between the two respective delay features.
