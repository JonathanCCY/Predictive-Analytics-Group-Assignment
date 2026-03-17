# Decision Log

## 2026-03-16 18:02:20 UTC - Step 1 benchmark bootstrap

- `agent_reproducibility_protocol_v3.md` was not found, so the user prompt is the active fallback specification.
- The source CSV files were present at the repository root instead of under `data/`.
- Step 1 created `data/` access paths for the existing source CSVs without generating the canonical split early.
- Step 1 seeded `docs/reproducibility_record.md` from the legacy root-level `reproducibility_record.md` file so later tasks can append in the expected location.
- `data/train.csv` preparation result: `created_symlink->../train.csv`.
- `data/test.csv` preparation result: `created_symlink->../test.csv`.

## 2026-03-16 18:11:12 UTC - Step 2 EDA and split creation

- `outputs/shared/split_manifest.json` was `created` in Step 2 using deterministic stratified `70/15/15` splitting with `random_state=42`.
- The manifest stores combined row indices so later tasks can reuse the exact same split without regenerating it.
- All target-aware EDA calculations and plots were restricted to the training split only.
- Univariate association ranking used point-biserial correlation for numeric features and Cramer's V for categorical features.
- Strongest training-split association signals observed: Class (cramers_v, 0.504), Online boarding (point_biserial, 0.501), Type of Travel (cramers_v, 0.449), Inflight entertainment (point_biserial, 0.400), Seat comfort (point_biserial, 0.349).
- Data quality flags carried forward for modelling review: Customer Type labels use mixed capitalization (for example `Loyal Customer` vs `disloyal Customer`).

## 2026-03-16 18:24:05 UTC - Step 3 model comparison and selection

- Reused `outputs/shared/split_manifest.json` without modification; SHA-256 at run time: `687510b33da154d45ae88e4425fadf5f43791b592bc8978b83a4b1b47a7b99f2`.
- The first in-sandbox execution hit an OpenMP shared-memory runtime error; the successful rerun was executed outside the sandbox with unchanged code, split, and model parameters.
- All four candidate models were trained on the training split only, with preprocessing fitted on training data only.
- Validation-only selection rule applied exactly as specified: ROC-AUC, then PR-AUC, then F1 at 0.5, then fixed priority order.
- Classification threshold remained fixed at `0.5`; no threshold tuning was attempted.
- Selected final model: `HistGradientBoostingClassifier` with validation ROC-AUC `0.994632`, PR-AUC `0.993664`, and F1 `0.956265`.
- Test metrics were written for all candidate models as post-selection descriptive comparison only.

## 2026-03-16 18:30:19 UTC - Step 4 broken pipeline fix

- Inspected the root-level `broken_pipeline.py` before making any fixes.
- Reused the canonical split from `outputs/shared/split_manifest.json` and verified it remained unchanged during the debug run.
- The first in-sandbox execution hit an OpenMP shared-memory runtime error; the successful rerun was executed outside the sandbox with unchanged split logic and LogisticRegression baseline intent.
- Preserved the original modelling intent as a LogisticRegression baseline, but corrected target handling, split reuse, train-only preprocessing, label alignment, and binary model persistence.
- Split manifest SHA-256 reused during the debug run: `687510b33da154d45ae88e4425fadf5f43791b592bc8978b83a4b1b47a7b99f2`.
- Debug outputs were saved under `outputs/debug/`.

## 2026-03-16 18:53:42 UTC - Step 5 documentation

- Wrote `README.md`, `docs/model_card.md`, and `docs/benchmark_summary.md` using repository-local evidence only.
- Documented the canonical split policy from `outputs/shared/split_manifest.json` and the selected-model evaluation from `outputs/model/`.
- Recorded `UNKNOWN` for runtime, token usage, monetary cost, and any standalone final-model-evaluation command because no separate checked-in entrypoint was found.
- Updated `docs/reproducibility_record.md` with a Step 5 evidence entry and completed the cross-task summary from saved artefacts.
- Noted the coexistence of a root-level `reproducibility_record.md` and `docs/reproducibility_record.md` as a documentation inconsistency for human review.
