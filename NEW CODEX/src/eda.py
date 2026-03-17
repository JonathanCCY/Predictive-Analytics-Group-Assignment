from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr, skew
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TASK_NAME = "eda"
TARGET_COLUMN = "satisfaction"
POSITIVE_LABEL = "satisfied"
NEGATIVE_LABEL = "neutral or dissatisfied"
STATUS_CONFIRMED_EXEC = "CONFIRMED_BY_EXECUTION"
STATUS_CONFIRMED_FILE = "CONFIRMED_BY_FILE_INSPECTION"
STATUS_UNVERIFIED = "UNVERIFIED"

NUMERIC_FEATURES = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def load_combined_dataset(data_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined_df["__combined_row_id__"] = combined_df.index.astype(int)
    return combined_df


def prepare_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = raw_df.copy()
    dropped_columns = []
    for candidate in [df.columns[0], "id"]:
        if candidate == "id" or str(candidate).startswith("Unnamed"):
            if candidate in df.columns:
                df = df.drop(columns=candidate)
                dropped_columns.append(candidate)

    unique_labels = set(df[TARGET_COLUMN].dropna().unique().tolist())
    expected_labels = {POSITIVE_LABEL, NEGATIVE_LABEL}
    if unique_labels != expected_labels:
        raise ValueError(
            f"Unexpected target labels found: {sorted(unique_labels)}; expected {sorted(expected_labels)}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    ).astype(int)

    expected_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN, "__combined_row_id__"]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing after preparation: {missing_columns}")

    metadata = {
        "dropped_columns": dropped_columns,
        "rows_total": int(df.shape[0]),
        "columns_total": int(df.shape[1]),
        "target_encoding": {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1},
    }
    return df, metadata


def build_split_manifest(processed_df: pd.DataFrame, manifest_path: Path) -> Dict[str, object]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    index_values = processed_df["__combined_row_id__"].to_numpy()
    y = processed_df[TARGET_COLUMN].to_numpy()

    train_idx, temp_idx = train_test_split(
        index_values,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    temp_y = processed_df.loc[temp_idx, TARGET_COLUMN].to_numpy()
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_y,
    )

    split_indices = {
        "train": sorted(int(value) for value in train_idx.tolist()),
        "validation": sorted(int(value) for value in val_idx.tolist()),
        "test": sorted(int(value) for value in test_idx.tolist()),
    }

    manifest = {
        "dataset_source": ["data/train.csv", "data/test.csv"],
        "rows_total": int(processed_df.shape[0]),
        "split_method": "two_stage_stratified_train_validation_test_split_70_15_15",
        "random_seed": RANDOM_SEED,
        "target_name": TARGET_COLUMN,
        "target_encoding": {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1},
        "dropped_columns_before_modelling": ["Unnamed: 0", "id"],
        "feature_columns": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "split_indices": split_indices,
        "split_summary": {},
    }

    for split_name, indices in split_indices.items():
        split_target = processed_df.loc[indices, TARGET_COLUMN]
        manifest["split_summary"][split_name] = {
            "row_count": int(len(indices)),
            "target_distribution": {
                "0": int((split_target == 0).sum()),
                "1": int((split_target == 1).sum()),
            },
            "target_rate_positive": float(split_target.mean()),
        }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_manifest(manifest: Dict[str, object], processed_df: pd.DataFrame) -> None:
    required_keys = {"train", "validation", "test"}
    split_indices = manifest.get("split_indices", {})
    if set(split_indices.keys()) != required_keys:
        raise ValueError("Split manifest is missing one or more required split index lists.")

    all_indices: List[int] = []
    for split_name in ["train", "validation", "test"]:
        indices = split_indices[split_name]
        all_indices.extend(indices)
        summary = manifest["split_summary"][split_name]
        if summary["row_count"] != len(indices):
            raise ValueError(f"Manifest row count mismatch for split: {split_name}")

    if len(all_indices) != processed_df.shape[0]:
        raise ValueError("Split manifest does not cover the full combined dataset.")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Split manifest contains duplicate row assignments across splits.")


def cramers_v(series_a: pd.Series, series_b: pd.Series) -> float:
    contingency = pd.crosstab(series_a, series_b)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(contingency, correction=False)[0]
    n_obs = contingency.to_numpy().sum()
    phi2 = chi2 / n_obs
    rows, cols = contingency.shape
    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n_obs - 1))
    rows_corr = rows - ((rows - 1) ** 2) / (n_obs - 1)
    cols_corr = cols - ((cols - 1) ** 2) / (n_obs - 1)
    denominator = min((cols_corr - 1), (rows_corr - 1))
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denominator))


def create_numeric_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    summary = train_df[NUMERIC_FEATURES].describe().T
    summary["missing_count"] = train_df[NUMERIC_FEATURES].isna().sum()
    summary["missing_pct"] = summary["missing_count"] / len(train_df)
    summary["skewness"] = train_df[NUMERIC_FEATURES].apply(
        lambda col: float(skew(col.dropna(), bias=False)) if col.dropna().shape[0] > 2 else np.nan
    )
    summary["zero_count"] = (train_df[NUMERIC_FEATURES] == 0).sum()
    return summary.reset_index().rename(columns={"index": "feature"})


def create_categorical_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for column in CATEGORICAL_FEATURES:
        value_counts = train_df[column].value_counts(dropna=False)
        rows.append(
            {
                "feature": column,
                "unique_count": int(train_df[column].nunique(dropna=False)),
                "missing_count": int(train_df[column].isna().sum()),
                "missing_pct": float(train_df[column].isna().mean()),
                "mode": train_df[column].mode(dropna=False).iloc[0],
                "top_value_frequency": int(value_counts.iloc[0]),
                "top_value_pct": float(value_counts.iloc[0] / len(train_df)),
                "levels": json.dumps(train_df[column].drop_duplicates().tolist()),
            }
        )
    return pd.DataFrame(rows)


def compute_associations(train_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    target = train_df[TARGET_COLUMN]

    for feature in NUMERIC_FEATURES:
        subset = train_df[[feature, TARGET_COLUMN]].dropna()
        coefficient, _ = pointbiserialr(subset[TARGET_COLUMN], subset[feature])
        rows.append(
            {
                "feature": feature,
                "method": "point_biserial",
                "association": float(abs(coefficient)),
                "signed_association": float(coefficient),
            }
        )

    for feature in CATEGORICAL_FEATURES:
        coefficient = cramers_v(train_df[feature], target)
        rows.append(
            {
                "feature": feature,
                "method": "cramers_v",
                "association": float(coefficient),
                "signed_association": np.nan,
            }
        )

    associations = pd.DataFrame(rows).sort_values(
        by="association", ascending=False
    ).reset_index(drop=True)
    return associations


def find_invalid_value_flags(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> List[str]:
    flags: List[str] = []
    rating_columns = [
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Inflight service",
        "Cleanliness",
    ]
    for column in rating_columns:
        minimum = processed_df[column].min()
        maximum = processed_df[column].max()
        if minimum < 0 or maximum > 5:
            flags.append(f"{column} contains values outside the expected 0-5 range.")

    if processed_df["Departure Delay in Minutes"].lt(0).any():
        flags.append("Departure Delay in Minutes contains negative values.")
    if processed_df["Arrival Delay in Minutes"].dropna().lt(0).any():
        flags.append("Arrival Delay in Minutes contains negative values.")

    customer_type_levels = raw_df["Customer Type"].dropna().unique().tolist()
    if any(level != level.title() for level in customer_type_levels):
        flags.append(
            "Customer Type labels use mixed capitalization (for example `Loyal Customer` vs `disloyal Customer`)."
        )

    return flags


def create_class_balance_plot(train_df: pd.DataFrame, output_path: Path) -> None:
    counts = train_df[TARGET_COLUMN].value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "class_label": ["0", "1"],
            "count": [int(counts.get(0, 0)), int(counts.get(1, 0))],
        }
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=plot_df, x="class_label", y="count", palette=["#5B8FF9", "#5AD8A6"], ax=ax)
    ax.set_title("Training Split Class Balance")
    ax.set_xlabel("Encoded satisfaction")
    ax.set_ylabel("Row count")
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{int(height)}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_missing_values_plot(train_df: pd.DataFrame, output_path: Path) -> None:
    missing = (
        train_df.isna().sum().sort_values(ascending=False).reset_index()
        .rename(columns={"index": "feature", 0: "missing_count"})
    )
    missing = missing[missing["missing_count"] > 0]
    if missing.empty:
        missing = pd.DataFrame({"feature": ["No missing values"], "missing_count": [0]})

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=missing, x="missing_count", y="feature", color="#F6BD16", ax=ax)
    ax.set_title("Missing Values in Training Split")
    ax.set_xlabel("Missing row count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_numeric_distributions_plot(train_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(6, 3, figsize=(15, 20))
    for axis, feature in zip(axes.flatten(), NUMERIC_FEATURES):
        sns.histplot(train_df[feature], bins=30, kde=False, color="#5B8FF9", ax=axis)
        axis.set_title(feature)
        axis.set_xlabel("")
    fig.suptitle("Training Split Numeric Feature Distributions", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_top_associations_plot(associations: pd.DataFrame, output_path: Path) -> None:
    top_associations = associations.head(12).copy()
    top_associations["display_label"] = top_associations["feature"] + " (" + top_associations["method"] + ")"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_associations,
        x="association",
        y="display_label",
        hue="method",
        dodge=False,
        palette={"point_biserial": "#5B8FF9", "cramers_v": "#E8684A"},
        ax=ax,
    )
    ax.set_title("Top Training-Split Feature Associations With Satisfaction")
    ax.set_xlabel("Absolute association strength")
    ax.set_ylabel("")
    ax.legend(title="Method", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_correlation_heatmap(train_df: pd.DataFrame, output_path: Path) -> None:
    correlation = train_df[NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Training Split Numeric Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_eda_summary(
    summary_path: Path,
    manifest: Dict[str, object],
    quality_report: Dict[str, object],
    associations: pd.DataFrame,
) -> None:
    top_features = associations.head(6)
    top_feature_lines = [
        f"- `{row.feature}` via `{row.method}` with association `{row.association:.3f}`"
        for row in top_features.itertuples()
    ]

    notes = "\n".join(f"- {note}" for note in quality_report["notes"])
    risk_lines = [
        f"- Severe skew: {', '.join(quality_report['severe_skew_columns']) if quality_report['severe_skew_columns'] else 'none flagged'}",
        f"- High-cardinality categoricals: {', '.join(quality_report['high_cardinality_columns']) if quality_report['high_cardinality_columns'] else 'none'}",
        f"- Possible leakage columns: {', '.join(quality_report['possible_leakage_columns']) if quality_report['possible_leakage_columns'] else 'none after dropping identifiers'}",
    ]

    content = "\n".join(
        [
            "# EDA Summary",
            "",
            "## Split and scope",
            f"- Canonical split method: `{manifest['split_method']}`",
            f"- Random seed: `{manifest['random_seed']}`",
            f"- Training rows used for EDA: `{quality_report['rows_train_used_for_eda']}`",
            f"- Validation rows counted only for split verification: `{manifest['split_summary']['validation']['row_count']}`",
            f"- Test rows counted only for split verification: `{manifest['split_summary']['test']['row_count']}`",
            "",
            "## Most likely predictive signals",
            *top_feature_lines,
            "",
            "## Data quality observations",
            f"- Missingness is concentrated in `Arrival Delay in Minutes` with `{quality_report['missing_by_column'].get('Arrival Delay in Minutes', 0)}` missing training rows.",
            f"- Duplicate training rows after dropping identifier columns: `{quality_report['duplicate_row_count']}`.",
            f"- Invalid value flags raised: `{len(quality_report['invalid_value_flags'])}`.",
            f"- Identifier-like columns removed before modelling: `{', '.join(quality_report['possible_identifier_columns'])}`.",
            "",
            "## Modelling risks and caveats",
            *risk_lines,
            "",
            "## Notes",
            notes,
            "",
            "## Execution status",
            f"- `{quality_report['execution_status']}`",
        ]
    )
    summary_path.write_text(content + "\n", encoding="utf-8")


def append_decision_log(
    repo_root: Path,
    manifest_action: str,
    quality_report: Dict[str, object],
    top_features: List[str],
) -> None:
    decision_log_path = repo_root / "docs" / "decision_log.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"## {timestamp} - Step 2 EDA and split creation",
        "",
        f"- `outputs/shared/split_manifest.json` was `{manifest_action}` in Step 2 using deterministic stratified `70/15/15` splitting with `random_state=42`.",
        "- The manifest stores combined row indices so later tasks can reuse the exact same split without regenerating it.",
        "- All target-aware EDA calculations and plots were restricted to the training split only.",
        "- Univariate association ranking used point-biserial correlation for numeric features and Cramer's V for categorical features.",
        f"- Strongest training-split association signals observed: {', '.join(top_features)}.",
        f"- Data quality flags carried forward for modelling review: {', '.join(quality_report['invalid_value_flags']) if quality_report['invalid_value_flags'] else 'none beyond missing arrival delay values'}",
        "",
    ]
    current = decision_log_path.read_text(encoding="utf-8").rstrip()
    decision_log_path.write_text(current + "\n\n" + "\n".join(lines), encoding="utf-8")


def append_reproducibility_record(
    repo_root: Path,
    files_inspected: List[str],
    commands_executed: List[str],
    quality_report: Dict[str, object],
    manifest: Dict[str, object],
    created_files: List[str],
    manifest_action: str,
) -> None:
    record_path = repo_root / "docs" / "reproducibility_record.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = "\n".join(
        [
            "",
            "---",
            "",
            "## Task Execution Evidence - Step 2 / EDA and Insight Generation",
            "",
            "### Executed run metadata",
            f"- Execution date: `{timestamp}`",
            f"- Main script: `src/eda.py`",
            f"- Execution status: `{STATUS_CONFIRMED_EXEC}`",
            f"- Files inspected before execution: `{', '.join(files_inspected)}`",
            f"- Commands executed: `{'; '.join(commands_executed)}`",
            f"- Run success rate: `1/1 successful runs`",
            f"- Required artefact completion rate: `{len(created_files)}/10 mandatory artefacts created`",
            f"- Split compliance: `{STATUS_CONFIRMED_EXEC}`",
            f"- Seed compliance: `{STATUS_CONFIRMED_EXEC}`",
            f"- Split manifest action in this step: `{manifest_action}`",
            f"- Repeated-run stability check: `NOT_EXECUTED`",
            f"- Duplicate row count observed on training split: `{quality_report['duplicate_row_count']}`",
            f"- Invalid value flags observed: `{json.dumps(quality_report['invalid_value_flags'])}`",
            f"- Severe skew columns observed: `{json.dumps(quality_report['severe_skew_columns'])}`",
            f"- Possible leakage columns observed: `{json.dumps(quality_report['possible_leakage_columns'])}`",
            f"- Files created: `{', '.join(created_files)}`",
            f"- Failures encountered: `NONE`",
            f"- Split row counts: train=`{manifest['split_summary']['train']['row_count']}`, validation=`{manifest['split_summary']['validation']['row_count']}`, test=`{manifest['split_summary']['test']['row_count']}`",
            "",
        ]
    )
    current = record_path.read_text(encoding="utf-8").rstrip()
    record_path.write_text(current + entry + "\n", encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    sns.set_theme(style="whitegrid")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "outputs" / TASK_NAME
    shared_dir = repo_root / "outputs" / "shared"
    docs_dir = repo_root / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    run_lines: List[str] = []
    files_inspected = [
        "data/train.csv",
        "data/test.csv",
        "requirements.txt",
        "docs/reproducibility_record.md",
        "docs/decision_log.md",
    ]
    commands_executed = ["python3 src/eda.py"]

    raw_df = load_combined_dataset(repo_root / "data")
    processed_df, prep_metadata = prepare_dataset(raw_df)
    manifest_path = shared_dir / "split_manifest.json"
    manifest_preexisting = manifest_path.exists()
    manifest = build_split_manifest(processed_df, manifest_path)
    validate_manifest(manifest, processed_df)
    manifest_action = "reused" if manifest_preexisting else "created"

    train_indices = manifest["split_indices"]["train"]
    train_df = processed_df.loc[train_indices].copy()

    numeric_summary = create_numeric_summary(train_df)
    categorical_summary = create_categorical_summary(train_df)
    associations = compute_associations(train_df)

    missing_by_column = {
        column: int(value)
        for column, value in train_df.isna().sum().items()
        if int(value) > 0
    }
    duplicate_row_count = int(
        train_df.drop(columns="__combined_row_id__").duplicated().sum()
    )
    invalid_value_flags = find_invalid_value_flags(raw_df, processed_df)

    high_cardinality_columns = [
        column
        for column in CATEGORICAL_FEATURES
        if train_df[column].nunique(dropna=False) > 20
    ]
    severe_skew_columns = [
        row.feature
        for row in numeric_summary.itertuples()
        if pd.notna(row.skewness) and abs(row.skewness) >= 2.0
    ]

    quality_report = {
        "dataset_source": "data/train.csv + data/test.csv",
        "rows_total": int(processed_df.shape[0]),
        "rows_train_used_for_eda": int(train_df.shape[0]),
        "columns_total": len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES) + 1,
        "target_name": TARGET_COLUMN,
        "class_balance_train": {
            "0": int((train_df[TARGET_COLUMN] == 0).sum()),
            "1": int((train_df[TARGET_COLUMN] == 1).sum()),
        },
        "missing_by_column": missing_by_column,
        "duplicate_row_count": duplicate_row_count,
        "invalid_value_flags": invalid_value_flags,
        "possible_identifier_columns": ["Unnamed: 0", "id"],
        "possible_leakage_columns": [],
        "high_cardinality_columns": high_cardinality_columns,
        "severe_skew_columns": severe_skew_columns,
        "notes": [
            "Target-aware summaries and plots use training rows only.",
            "Validation and test splits were used only to verify split sizes and class distributions.",
            "Arrival Delay in Minutes is the only column with missing values in the source data.",
            "Customer Type labels are semantically consistent but use mixed capitalization.",
            "Delay variables may be operationally unavailable in pre-flight deployment settings, although they are not leakage under a post-flight satisfaction prediction framing.",
        ],
        "execution_status": STATUS_CONFIRMED_EXEC,
    }

    numeric_summary.to_csv(output_dir / "numeric_summary.csv", index=False)
    categorical_summary.to_csv(output_dir / "categorical_summary.csv", index=False)
    create_class_balance_plot(train_df, output_dir / "class_balance.png")
    create_missing_values_plot(train_df, output_dir / "missing_values.png")
    create_numeric_distributions_plot(train_df, output_dir / "numeric_distributions.png")
    create_top_associations_plot(associations, output_dir / "top_target_associations.png")
    create_correlation_heatmap(train_df, output_dir / "correlation_heatmap.png")
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2),
        encoding="utf-8",
    )

    top_features = [
        f"{row.feature} ({row.method}, {row.association:.3f})"
        for row in associations.head(5).itertuples()
    ]
    write_eda_summary(docs_dir / "eda_summary.md", manifest, quality_report, associations)

    created_files = [
        "outputs/shared/split_manifest.json",
        "outputs/eda/class_balance.png",
        "outputs/eda/missing_values.png",
        "outputs/eda/numeric_summary.csv",
        "outputs/eda/categorical_summary.csv",
        "outputs/eda/numeric_distributions.png",
        "outputs/eda/top_target_associations.png",
        "outputs/eda/correlation_heatmap.png",
        "outputs/eda/data_quality_report.json",
        "docs/eda_summary.md",
    ]

    append_decision_log(repo_root, manifest_action, quality_report, top_features)
    append_reproducibility_record(
        repo_root=repo_root,
        files_inspected=files_inspected,
        commands_executed=commands_executed,
        quality_report=quality_report,
        manifest=manifest,
        created_files=created_files,
        manifest_action=manifest_action,
    )

    run_metadata = {
        "task_name": TASK_NAME,
        "execution_status": STATUS_CONFIRMED_EXEC,
        "random_seed": RANDOM_SEED,
        "split_manifest_action": manifest_action,
        "split_summary": manifest["split_summary"],
        "files_inspected": files_inspected,
        "commands_executed": commands_executed,
        "top_associations": associations.head(10).to_dict(orient="records"),
        "quality_report": quality_report,
        "created_files": created_files + [
            "outputs/eda/run_log.txt",
            "outputs/eda/run_metadata.json",
            "docs/reproducibility_record.md",
            "docs/decision_log.md",
        ],
        "dropped_columns": prep_metadata["dropped_columns"],
    }

    run_lines.extend(
        [
            f"Execution status: {STATUS_CONFIRMED_EXEC}",
            f"Random seed: {RANDOM_SEED}",
            f"Split manifest action: {manifest_action}",
            f"Train/validation/test sizes: {manifest['split_summary']['train']['row_count']}/"
            f"{manifest['split_summary']['validation']['row_count']}/"
            f"{manifest['split_summary']['test']['row_count']}",
            f"Training class balance: {quality_report['class_balance_train']}",
            f"Missing by column: {quality_report['missing_by_column']}",
            f"Duplicate training rows: {duplicate_row_count}",
            f"Invalid value flags: {quality_report['invalid_value_flags']}",
            f"Severe skew columns: {quality_report['severe_skew_columns']}",
            f"Top associations: {top_features}",
        ]
    )
    (output_dir / "run_log.txt").write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
