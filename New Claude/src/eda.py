"""
Step 2: EDA and Insight Generation
- Combines train/test CSVs, drops unnamed index + id
- Creates canonical stratified 70/15/15 split (seed=42)
- Performs target-aware EDA on training rows only
- Saves all mandatory artefacts
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT_EDA = os.path.join(ROOT, 'outputs', 'eda')
OUT_SHARED = os.path.join(ROOT, 'outputs', 'shared')

os.makedirs(OUT_EDA, exist_ok=True)
os.makedirs(OUT_SHARED, exist_ok=True)

SEED = 42
np.random.seed(SEED)

run_log_lines = []

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    run_log_lines.append(line)


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD AND COMBINE DATA
# ══════════════════════════════════════════════════════════════════════════
log("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
log(f"  train.csv: {train_df.shape}, test.csv: {test_df.shape}")

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
log(f"  Combined: {df.shape}")

# Drop unnamed first column (row index from Kaggle export)
unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)
    log(f"  Dropped unnamed columns: {unnamed_cols}")

# Drop id column
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
    log("  Dropped 'id' column")

log(f"  Final shape after drops: {df.shape}")
log(f"  Columns ({len(df.columns)}): {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════
# 2. ENCODE TARGET
# ══════════════════════════════════════════════════════════════════════════
log("Encoding target variable...")
log(f"  Original target values: {df['satisfaction'].unique()}")
df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
assert df['satisfaction'].isna().sum() == 0, "Unmapped target values found!"
log(f"  Encoded target distribution:\n{df['satisfaction'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════════════════════
# 3. CREATE CANONICAL STRATIFIED SPLIT (70/15/15)
# ══════════════════════════════════════════════════════════════════════════
log("Creating stratified 70/15/15 split...")
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)

# Second split: 50/50 of temp -> 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

log(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
log(f"  Total: {len(y_train) + len(y_val) + len(y_test)}")

# Save split manifest
split_manifest = {
    "description": "Canonical stratified 70/15/15 split for Airline Passenger Satisfaction benchmark",
    "random_seed": SEED,
    "split_method": "sklearn.model_selection.train_test_split (two-stage: 70/30 then 50/50 of remainder)",
    "stratify_column": "satisfaction",
    "source_files": ["data/train.csv", "data/test.csv"],
    "combined_rows_before_split": len(df),
    "columns_dropped_before_split": ["Unnamed: 0", "id"],
    "splits": {
        "train": {
            "n_rows": int(len(y_train)),
            "target_distribution": {
                "0": int((y_train == 0).sum()),
                "1": int((y_train == 1).sum())
            },
            "indices": sorted(X_train.index.tolist())
        },
        "validation": {
            "n_rows": int(len(y_val)),
            "target_distribution": {
                "0": int((y_val == 0).sum()),
                "1": int((y_val == 1).sum())
            },
            "indices": sorted(X_val.index.tolist())
        },
        "test": {
            "n_rows": int(len(y_test)),
            "target_distribution": {
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum())
            },
            "indices": sorted(X_test.index.tolist())
        }
    }
}

manifest_path = os.path.join(OUT_SHARED, 'split_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(split_manifest, f, indent=2)
log(f"  Split manifest saved to {manifest_path}")

# ══════════════════════════════════════════════════════════════════════════
# 4. PREPARE TRAINING DATA FOR EDA
# ══════════════════════════════════════════════════════════════════════════
train_data = df.loc[X_train.index].copy()
log(f"Training data for EDA: {train_data.shape}")

numeric_features = [
    'Age', 'Flight Distance',
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# ══════════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
log("Analysing data quality...")

# Missing values (on full combined df, then train)
missing_all = df.isnull().sum()
missing_train = train_data.isnull().sum()
log(f"  Missing values in training set:\n{missing_train[missing_train > 0].to_string()}")

# Duplicates
dup_count = train_data.drop(columns=['satisfaction']).duplicated().sum()
log(f"  Duplicate rows (excl. target) in training set: {dup_count}")

# Invalid values: check for negative delays
invalid_flags = []
for col in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    neg_count = (train_data[col] < 0).sum()
    if neg_count > 0:
        invalid_flags.append(f"{col}: {neg_count} negative values")

# Check rating columns (should be 0-5)
rating_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]
for col in rating_cols:
    out_of_range = ((train_data[col] < 0) | (train_data[col] > 5)).sum()
    if out_of_range > 0:
        invalid_flags.append(f"{col}: {out_of_range} values outside [0,5]")

if not invalid_flags:
    invalid_flags.append("No invalid values detected")
log(f"  Invalid value flags: {invalid_flags}")

# Severe skew (>2 or <-2)
skew_values = train_data[numeric_features].skew()
severe_skew_cols = skew_values[skew_values.abs() > 2].index.tolist()
log(f"  Severe skew columns (|skew|>2): {severe_skew_cols}")

# Possible leakage: Arrival Delay closely mirrors Departure Delay
corr_delays = train_data['Departure Delay in Minutes'].corr(
    train_data['Arrival Delay in Minutes']
)
possible_leakage = []
if corr_delays > 0.9:
    possible_leakage.append(
        f"Arrival Delay in Minutes (r={corr_delays:.3f} with Departure Delay in Minutes — near-duplicate information)"
    )
log(f"  Possible leakage columns: {possible_leakage}")

# High cardinality check
high_card = []
for col in categorical_features:
    n_unique = train_data[col].nunique()
    if n_unique > 20:
        high_card.append(f"{col}: {n_unique} unique values")
log(f"  High cardinality columns: {high_card if high_card else 'None'}")

# ══════════════════════════════════════════════════════════════════════════
# 6. SAVE DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════
data_quality_report = {
    "dataset_source": "data/train.csv + data/test.csv (Kaggle Airline Passenger Satisfaction)",
    "rows_total": int(len(df)),
    "rows_train_used_for_eda": int(len(train_data)),
    "columns_total": int(len(df.columns)),
    "target_name": "satisfaction",
    "class_balance_train": {
        "0": int((train_data['satisfaction'] == 0).sum()),
        "1": int((train_data['satisfaction'] == 1).sum())
    },
    "missing_by_column": {col: int(v) for col, v in missing_train.items() if v > 0},
    "duplicate_row_count": int(dup_count),
    "invalid_value_flags": invalid_flags,
    "possible_identifier_columns": ["id (dropped)", "Unnamed: 0 (dropped)"],
    "possible_leakage_columns": possible_leakage if possible_leakage else ["None identified"],
    "high_cardinality_columns": high_card if high_card else ["None identified"],
    "severe_skew_columns": severe_skew_cols if severe_skew_cols else ["None identified"],
    "notes": [
        f"Arrival Delay in Minutes has {int(missing_train.get('Arrival Delay in Minutes', 0))} missing values in training set",
        "Rating columns (14 service features) use integer scale 0-5",
        "Delay columns are right-skewed with long tails",
        f"Correlation between Departure and Arrival Delay: {corr_delays:.3f}"
    ],
    "execution_status": "CONFIRMED_BY_EXECUTION"
}

with open(os.path.join(OUT_EDA, 'data_quality_report.json'), 'w') as f:
    json.dump(data_quality_report, f, indent=2)
log("Data quality report saved.")

# ══════════════════════════════════════════════════════════════════════════
# 7. NUMERIC SUMMARY
# ══════════════════════════════════════════════════════════════════════════
log("Generating numeric summary...")
numeric_summary = train_data[numeric_features].describe().T
numeric_summary['missing'] = train_data[numeric_features].isnull().sum()
numeric_summary['missing_pct'] = (numeric_summary['missing'] / len(train_data) * 100).round(2)
numeric_summary['skew'] = train_data[numeric_features].skew()
numeric_summary['kurtosis'] = train_data[numeric_features].kurtosis()
numeric_summary.to_csv(os.path.join(OUT_EDA, 'numeric_summary.csv'))
log("  numeric_summary.csv saved.")

# ══════════════════════════════════════════════════════════════════════════
# 8. CATEGORICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
log("Generating categorical summary...")
cat_rows = []
for col in categorical_features:
    vc = train_data[col].value_counts()
    for val, cnt in vc.items():
        cat_rows.append({
            'feature': col,
            'value': val,
            'count': cnt,
            'proportion': round(cnt / len(train_data), 4),
            'satisfaction_rate': round(
                train_data.loc[train_data[col] == val, 'satisfaction'].mean(), 4
            )
        })
cat_summary = pd.DataFrame(cat_rows)
cat_summary.to_csv(os.path.join(OUT_EDA, 'categorical_summary.csv'), index=False)
log("  categorical_summary.csv saved.")

# ══════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({'figure.dpi': 120, 'savefig.bbox': 'tight'})

# ── 9a. Class balance ──
log("Plotting class balance...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
class_counts = train_data['satisfaction'].value_counts().sort_index()
labels = ['Neutral/Dissatisfied (0)', 'Satisfied (1)']
colors = ['#e74c3c', '#2ecc71']

axes[0].bar(labels, class_counts.values, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
axes[0].set_title('Class Distribution (Training Set)')
axes[0].set_ylabel('Count')

axes[1].pie(class_counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
axes[1].set_title('Class Proportions')

plt.tight_layout()
plt.savefig(os.path.join(OUT_EDA, 'class_balance.png'))
plt.close()
log("  class_balance.png saved.")

# ── 9b. Missing values ──
log("Plotting missing values...")
fig, ax = plt.subplots(figsize=(10, 5))
missing_pct = (train_data.isnull().sum() / len(train_data) * 100).sort_values(ascending=True)
# Show all columns, highlight those with missing
bar_colors = ['#e74c3c' if v > 0 else '#95a5a6' for v in missing_pct.values]
missing_pct.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='black', linewidth=0.3)
ax.set_xlabel('Missing (%)')
ax.set_title('Missing Values by Column (Training Set)')
ax.axvline(x=0, color='black', linewidth=0.5)
for i, (col, val) in enumerate(missing_pct.items()):
    if val > 0:
        ax.text(val + 0.02, i, f'{val:.2f}%', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_EDA, 'missing_values.png'))
plt.close()
log("  missing_values.png saved.")

# ── 9c. Numeric distributions ──
log("Plotting numeric distributions...")
n_cols = 4
n_rows = (len(numeric_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    ax = axes[i]
    train_data[col].dropna().hist(bins=30, ax=ax, color='#3498db', edgecolor='black',
                                   linewidth=0.3, alpha=0.8)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

# Hide unused subplots
for j in range(len(numeric_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Numeric Feature Distributions (Training Set)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_EDA, 'numeric_distributions.png'))
plt.close()
log("  numeric_distributions.png saved.")

# ── 9d. Correlation heatmap ──
log("Plotting correlation heatmap...")
corr_matrix = train_data[numeric_features + ['satisfaction']].corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 7},
            vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap (Training Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_EDA, 'correlation_heatmap.png'))
plt.close()
log("  correlation_heatmap.png saved.")

# ── 9e. Top target associations ──
log("Computing target associations...")

# Point-biserial correlation for numeric features
pb_corrs = {}
for col in numeric_features:
    mask_valid = train_data[col].notna()
    r, p = stats.pointbiserialr(train_data.loc[mask_valid, 'satisfaction'],
                                 train_data.loc[mask_valid, col])
    pb_corrs[col] = abs(r)

# Cramér's V for categorical features
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0

cat_assoc = {}
for col in categorical_features:
    cat_assoc[col] = cramers_v(train_data[col], train_data['satisfaction'])

# Combine and sort
all_assoc = {}
for col, val in pb_corrs.items():
    all_assoc[col] = ('Point-biserial |r|', val)
for col, val in cat_assoc.items():
    all_assoc[col] = ("Cramér's V", val)

assoc_df = pd.DataFrame([
    {'feature': k, 'method': v[0], 'association': v[1]}
    for k, v in all_assoc.items()
]).sort_values('association', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bar_colors = ['#e67e22' if m == "Cramér's V" else '#3498db'
              for m in assoc_df['method']]
bars = ax.barh(assoc_df['feature'], assoc_df['association'], color=bar_colors,
               edgecolor='black', linewidth=0.3)
ax.set_xlabel('Association Strength')
ax.set_title('Feature–Target Associations (Training Set)\nNumeric: Point-biserial |r| (blue) | Categorical: Cramér\'s V (orange)',
             fontsize=11)
for i, (feat, val) in enumerate(zip(assoc_df['feature'], assoc_df['association'])):
    ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_EDA, 'top_target_associations.png'))
plt.close()
log("  top_target_associations.png saved.")

# ── 9f. Categorical vs target (recommended extra) ──
log("Plotting categorical vs target...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    ax = axes[i]
    ct = train_data.groupby([col, 'satisfaction']).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    ct_pct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{col} vs Satisfaction', fontsize=10, fontweight='bold')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('')
    ax.legend(['Neutral/Dissatisfied', 'Satisfied'], fontsize=7, loc='upper right')
    ax.tick_params(axis='x', rotation=0, labelsize=8)
fig.suptitle('Categorical Features vs Target (Training Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_EDA, 'categorical_vs_target.png'))
plt.close()
log("  categorical_vs_target.png saved.")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE RUN LOG AND METADATA
# ══════════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT_EDA, 'run_log.txt'), 'w') as f:
    f.write('\n'.join(run_log_lines))

run_metadata = {
    "task": "Step 2 - EDA and Insight Generation",
    "script": "src/eda.py",
    "execution_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "python_version": sys.version,
    "random_seed": SEED,
    "split_ratios": "70/15/15",
    "train_rows": int(len(train_data)),
    "val_rows": int(len(y_val)),
    "test_rows": int(len(y_test)),
    "eda_scope": "training rows only",
    "execution_status": "CONFIRMED_BY_EXECUTION"
}
with open(os.path.join(OUT_EDA, 'run_metadata.json'), 'w') as f:
    json.dump(run_metadata, f, indent=2)

log("EDA script completed successfully.")
log(f"Total artefacts directory: {OUT_EDA}")
