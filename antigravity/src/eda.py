import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from sklearn.model_selection import train_test_split
from scipy.stats import pointbiserialr, chi2_contingency

def main():
    try:
        # Paths
        DATA_DIR = 'data'
        OUTPUT_SHARED_DIR = 'outputs/shared'
        OUTPUT_EDA_DIR = 'outputs/eda'
        
        os.makedirs(OUTPUT_SHARED_DIR, exist_ok=True)
        os.makedirs(OUTPUT_EDA_DIR, exist_ok=True)
        
        # 1. Load and combine data
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        rows_total_before = len(df)
        cols_total_before = df.shape[1]
        
        # 2. Drop specific columns
        # Drop the unnamed first column and id
        drop_cols = []
        for col in df.columns[:5]: 
            if 'Unnamed' in col or col.lower() == 'id':
                drop_cols.append(col)
                
        if 'id' in df.columns and 'id' not in drop_cols:
            drop_cols.append('id')
            
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        cols_total_after = df.shape[1]
        
        # 3. Target encoding
        target_col = 'satisfaction'
        df[target_col] = df[target_col].map({'satisfied': 1, 'neutral or dissatisfied': 0})
        if df[target_col].isnull().any():
            print("WARNING: Some target values could not be mapped!")
        
        # 4. Stratified Split 70/15/15
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        # 15% validation, 15% test (50% of temp each)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
        
        # Save split_manifest.json
        split_manifest = {
            "random_seed": 42,
            "split_method": "sklearn.model_selection.train_test_split stratified on target. 70/15/15 across train/val/test.",
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "row_counts": {
                "train": len(X_train),
                "validation": len(X_val),
                "test": len(X_test),
                "total": len(df)
            },
            "target_distribution": {
                "train": y_train.value_counts(dropna=False).to_dict(),
                "validation": y_val.value_counts(dropna=False).to_dict(),
                "test": y_test.value_counts(dropna=False).to_dict()
            }
        }
        with open(os.path.join(OUTPUT_SHARED_DIR, 'split_manifest.json'), 'w') as f:
            json.dump(split_manifest, f, indent=4)
        
        # Recombine X and y for EDA (TRAIN ONLY)
        eda_df = pd.concat([X_train, y_train], axis=1)
        
        # Data Quality Checks on Train
        duplicates = int(eda_df.duplicated().sum())
        missing_by_col = eda_df.isnull().sum()
        missing_dict = missing_by_col[missing_by_col > 0].to_dict()
        
        num_cols = eda_df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()
        cat_cols = eda_df.select_dtypes(exclude=[np.number]).columns.drop(target_col, errors='ignore').tolist()
        
        severe_skew = []
        for col in num_cols:
            if abs(eda_df[col].skew()) > 3:
                severe_skew.append(col)
                
        high_cardinality = []
        for col in cat_cols:
            if eda_df[col].nunique() > 20:
                high_cardinality.append(col)
                
        # 5. Visualisations & Artifacts
        # class_balance.png
        plt.figure(figsize=(8,6))
        sns.countplot(x=target_col, data=eda_df)
        plt.title('Target Class Balance (Train)')
        plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'class_balance.png'), bbox_inches='tight')
        plt.close()
        
        # missing_values.png
        plt.figure(figsize=(10,6))
        missing_pct = (missing_by_col / len(eda_df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if len(missing_pct) > 0:
            sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='viridis')
            plt.title('Percentage of Missing Values (Train)')
            plt.xlabel('% Missing')
            plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'missing_values.png'), bbox_inches='tight')
        else:
            plt.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
            plt.title('Percentage of Missing Values (Train)')
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'missing_values.png'), bbox_inches='tight')
        plt.close()
        
        # numeric_summary.csv & categorical_summary.csv
        if len(num_cols) > 0:
            eda_df[num_cols].describe().T.to_csv(os.path.join(OUTPUT_EDA_DIR, 'numeric_summary.csv'))
        else:
            pd.DataFrame().to_csv(os.path.join(OUTPUT_EDA_DIR, 'numeric_summary.csv'))
            
        if len(cat_cols) > 0:
            eda_df[cat_cols].describe(include='O').T.to_csv(os.path.join(OUTPUT_EDA_DIR, 'categorical_summary.csv'))
        else:
            pd.DataFrame().to_csv(os.path.join(OUTPUT_EDA_DIR, 'categorical_summary.csv'))
            
        # numeric_distributions.png
        n_num = len(num_cols)
        if n_num > 0:
            cols_subplot = 4
            rows_subplot = (n_num + cols_subplot - 1) // cols_subplot
            fig, axes = plt.subplots(ncols=cols_subplot, nrows=rows_subplot, figsize=(20, 5 * rows_subplot))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                sns.histplot(eda_df[col].dropna(), ax=axes[i], kde=True, bins=30)
                axes[i].set_title(col[:30])
            for i in range(n_num, len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'numeric_distributions.png'))
            plt.close()
        else:
            plt.figure()
            plt.text(0.5, 0.5, 'No numerical columns')
            plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'numeric_distributions.png'))
            plt.close()
            
        # correlation_heatmap.png
        plt.figure(figsize=(16, 12))
        corr_matrix = eda_df[num_cols + [target_col]].dropna().corr(method='spearman')
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap (Spearman)')
        plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'correlation_heatmap.png'), bbox_inches='tight')
        plt.close()
        
        # top_target_associations.png
        associations = {}
        # Numeric using absolute point-biserial correlation
        for col in num_cols:
            clean_df = eda_df[[col, target_col]].dropna()
            if len(clean_df) > 0:
                corr, _ = pointbiserialr(clean_df[target_col], clean_df[col])
                associations[col] = abs(corr) if not np.isnan(corr) else 0.0
                
        # Categorical using Cramér's V
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            d = min((kcorr-1), (rcorr-1))
            if d == 0:
                return 0.0
            return np.sqrt(phi2corr / d)

        for col in cat_cols:
            clean_df = eda_df[[col, target_col]].dropna()
            if len(clean_df) > 0:
                v = cramers_v(clean_df[col], clean_df[target_col])
                associations[col] = v
                
        assoc_df = pd.Series(associations).sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(10,8))
        sns.barplot(x=assoc_df.values, y=assoc_df.index, palette='magma')
        plt.title('Top Target Associations (Abs Point-Biserial for Num, Cramers V for Cat)')
        plt.xlabel('Association Score')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_EDA_DIR, 'top_target_associations.png'))
        plt.close()
        
        # Data Quality Report JSON
        dq_report = {
          "dataset_source": "data/train.csv and data/test.csv combined before split",
          "rows_total": rows_total_before,
          "rows_train_used_for_eda": len(eda_df),
          "columns_total": cols_total_after + 1,
          "target_name": target_col,
          "class_balance_train": {str(int(k)): int(v) for k,v in eda_df[target_col].value_counts().to_dict().items()},
          "missing_by_column": {k: int(v) for k,v in missing_dict.items()},
          "duplicate_row_count": duplicates,
          "invalid_value_flags": ["Missing values found in Arrival Delay in Minutes"] if "Arrival Delay in Minutes" in missing_dict else [],
          "possible_identifier_columns": [],
          "possible_leakage_columns": [], # None obvious immediately, maybe delays? Need manual interpretation in markdown
          "high_cardinality_columns": high_cardinality,
          "severe_skew_columns": severe_skew,
          "notes": ["EDA executed strictly on the 70% training split only"],
          "execution_status": "CONFIRMED_BY_EXECUTION"
        }
        
        with open(os.path.join(OUTPUT_EDA_DIR, 'data_quality_report.json'), 'w') as f:
            json.dump(dq_report, f, indent=4)
            
        # Metadata and Log
        run_metadata = {"task": "eda", "status": "success", "seed_used": 42}
        with open(os.path.join(OUTPUT_EDA_DIR, 'run_metadata.json'), 'w') as f:
            json.dump(run_metadata, f, indent=4)
            
        with open(os.path.join(OUTPUT_EDA_DIR, 'run_log.txt'), 'w') as f:
            f.write("EDA completed successfully. Split created and artifacts saved.\n")
            
        print("EDA script finished successfully!")
        
    except Exception as e:
        print(f"Error during EDA: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
