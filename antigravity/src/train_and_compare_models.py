import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    try:
        # Paths
        DATA_DIR = 'data'
        OUTPUT_SHARED_DIR = 'outputs/shared'
        OUTPUT_COMPARE_DIR = 'outputs/model_compare'
        OUTPUT_MODEL_DIR = 'outputs/model'
        
        os.makedirs(OUTPUT_COMPARE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        
        # 1. Load Data
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Drop columns
        drop_cols = [c for c in df.columns[:5] if 'Unnamed' in c or c.lower() == 'id']
        if 'id' in df.columns and 'id' not in drop_cols:
            drop_cols.append('id')
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # Target encoding
        target_col = 'satisfaction'
        df[target_col] = df[target_col].map({'satisfied': 1, 'neutral or dissatisfied': 0})
        
        # 2. Re-apply exact split from manifest
        manifest_path = os.path.join(OUTPUT_SHARED_DIR, 'split_manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError("split_manifest.json not found! Must run Step 2 first.")
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        random_seed = manifest['random_seed']
        if random_seed != 42:
            raise ValueError(f"Expected random seed 42, got {random_seed}")
            
        # Reconstruct split exactly as done in Step 2:
        # train_test_split(df, test_size=0.30, random_state=42, stratify=y)
        # train_test_split(temp, test_size=0.50, random_state=42, stratify=y_temp)
        from sklearn.model_selection import train_test_split
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
        
        # Verification check
        if len(X_train) != manifest['row_counts']['train'] or len(X_val) != manifest['row_counts']['validation']:
            raise ValueError("Split reconstruction failed. Counts do not match manifest.")
            
        # Feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        with open(os.path.join(OUTPUT_MODEL_DIR, 'feature_manifest.json'), 'w') as f:
            json.dump({"numeric_features": numeric_features, "categorical_features": categorical_features}, f, indent=4)
            
        # 3. Model Definitions & Pipelines
        models_dict = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=100, random_state=42)
        }
        
        pipelines = {}
        for name, model in models_dict.items():
            if name == "LogisticRegression":
                num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            elif name in ["RandomForestClassifier", "ExtraTreesClassifier"]:
                num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            else: # HistGradientBoostingClassifier
                num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
                
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, numeric_features),
                    ('cat', cat_pipe, categorical_features)
                ])
                
            pipelines[name] = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
            
        # 4. Training and Evaluation
        def eval_model(model_obj, X_eval, y_eval, threshold=0.5):
            probs = model_obj.predict_proba(X_eval)[:, 1]
            preds = (probs >= threshold).astype(int)
            return {
                "ROC-AUC": roc_auc_score(y_eval, probs),
                "PR-AUC": average_precision_score(y_eval, probs),
                "Accuracy": accuracy_score(y_eval, preds),
                "Precision": precision_score(y_eval, preds, zero_division=0),
                "Recall": recall_score(y_eval, preds, zero_division=0),
                "F1-score": f1_score(y_eval, preds, zero_division=0)
            }, probs, preds, confusion_matrix(y_eval, preds)
            
        val_metrics_list = []
        test_metrics_list = []
        
        trained_models = {}
        val_probs_dict = {}
        val_preds_dict = {}
        val_cm_dict = {}
        
        test_probs_dict = {}
        test_preds_dict = {}
        test_cm_dict = {}
        
        for name, pipe in pipelines.items():
            print(f"Training {name}...")
            pipe.fit(X_train, y_train)
            trained_models[name] = pipe
            
            val_metrics, v_probs, v_preds, v_cm = eval_model(pipe, X_val, y_val)
            val_metrics['Model'] = name
            val_metrics_list.append(val_metrics)
            val_probs_dict[name] = v_probs
            val_preds_dict[name] = v_preds
            val_cm_dict[name] = v_cm
            
            test_metrics, t_probs, t_preds, t_cm = eval_model(pipe, X_test, y_test)
            test_metrics['Model'] = name
            test_metrics_list.append(test_metrics)
            test_probs_dict[name] = t_probs
            test_preds_dict[name] = t_preds
            test_cm_dict[name] = t_cm
            
        val_df = pd.DataFrame(val_metrics_list).set_index('Model')
        test_df = pd.DataFrame(test_metrics_list).set_index('Model')
        
        val_df.to_csv(os.path.join(OUTPUT_COMPARE_DIR, 'validation_metrics_by_model.csv'))
        test_df.to_csv(os.path.join(OUTPUT_COMPARE_DIR, 'test_metrics_by_model.csv'))
        
        # 5. Model Selection
        # primary: validation ROC-AUC
        # tie 1: validation PR-AUC
        # tie 2: validation F1-score
        # tie 3: priority order
        
        priority_order = {"LogisticRegression": 1, "HistGradientBoostingClassifier": 2, "RandomForestClassifier": 3, "ExtraTreesClassifier": 4}
        val_df['priority'] = val_df.index.map(priority_order)
        
        val_df_sorted = val_df.sort_values(by=['ROC-AUC', 'PR-AUC', 'F1-score', 'priority'], ascending=[False, False, False, True])
        
        best_model_name = val_df_sorted.index[0]
        print(f"Selected Best Model: {best_model_name}")
        
        # 6. Save Best Model Outputs
        best_model = trained_models[best_model_name]
        joblib.dump(best_model, os.path.join(OUTPUT_MODEL_DIR, 'model.joblib'))
        
        pd.DataFrame({"probability": val_probs_dict[best_model_name], "prediction": val_preds_dict[best_model_name]}).to_csv(
            os.path.join(OUTPUT_MODEL_DIR, 'validation_predictions.csv'), index=False)
            
        pd.DataFrame({"probability": test_probs_dict[best_model_name], "prediction": test_preds_dict[best_model_name]}).to_csv(
            os.path.join(OUTPUT_MODEL_DIR, 'test_predictions.csv'), index=False)
            
        with open(os.path.join(OUTPUT_MODEL_DIR, 'metrics_validation.json'), 'w') as f:
            json.dump(val_df.loc[best_model_name].drop('priority').to_dict(), f, indent=4)
            
        with open(os.path.join(OUTPUT_MODEL_DIR, 'metrics_test.json'), 'w') as f:
            json.dump(test_df.loc[best_model_name].to_dict(), f, indent=4)
            
        pd.DataFrame(val_cm_dict[best_model_name]).to_csv(os.path.join(OUTPUT_MODEL_DIR, 'confusion_matrix_validation.csv'), index=False)
        pd.DataFrame(test_cm_dict[best_model_name]).to_csv(os.path.join(OUTPUT_MODEL_DIR, 'confusion_matrix_test.csv'), index=False)
        
        # 7. Model Compare Manifest and Report
        manifest_cmp = {
          "candidate_models": ["LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier", "HistGradientBoostingClassifier"],
          "preprocessing_by_model": {
            "LogisticRegression": "median impute + StandardScaler for numeric; most_frequent impute + OneHotEncoder for categorical",
            "RandomForestClassifier": "median impute for numeric; most_frequent impute + OneHotEncoder for categorical",
            "ExtraTreesClassifier": "median impute for numeric; most_frequent impute + OneHotEncoder for categorical",
            "HistGradientBoostingClassifier": "median impute for numeric; most_frequent impute + OrdinalEncoder for categorical"
          },
          "fixed_parameters_by_model": {
            "LogisticRegression": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"},
            "RandomForestClassifier": {"n_estimators": 100, "random_state": 42},
            "ExtraTreesClassifier": {"n_estimators": 100, "random_state": 42},
            "HistGradientBoostingClassifier": {"max_iter": 100, "random_state": 42}
          },
          "split_manifest_path": "outputs/shared/split_manifest.json",
          "target_name": "satisfaction",
          "random_seed": 42,
          "selection_metric": "validation ROC-AUC",
          "tie_break_rules": ["PR-AUC", "F1 at 0.5", "fixed priority order"],
          "threshold_rule": "fixed at 0.5",
          "dependency_scope": "requirements.txt",
          "execution_status": "CONFIRMED_BY_EXECUTION"
        }
        
        with open(os.path.join(OUTPUT_COMPARE_DIR, 'candidate_model_manifest.json'), 'w') as f:
            json.dump(manifest_cmp, f, indent=4)
            
        report_md = f"""# Model Selection Report

## Context
Four baseline candidate models were trained strictly according to the benchmark specifications to predict passenger `satisfaction`.

## Models Evaluated
1. LogisticRegression
2. RandomForestClassifier
3. ExtraTreesClassifier
4. HistGradientBoostingClassifier

## Selection Mechanism
Models were evaluated exclusively on the validation set.
- Primary standard: **ROC-AUC**
- Tie 1: PR-AUC
- Tie 2: F1-score at 0.5 threshold

### Results (Validation)
{val_df.drop(columns=['priority']).to_markdown()}

## Conclusion
The model chosen for final evaluation on the test set is **{best_model_name}**.
All post-selection metrics on the test set for all models are located in `outputs/model_compare/test_metrics_by_model.csv` strictly as an observational comparison without exerting selection influence.
"""
        with open(os.path.join(OUTPUT_COMPARE_DIR, 'model_selection_report.md'), 'w') as f:
            f.write(report_md)
            
        with open(os.path.join(OUTPUT_COMPARE_DIR, 'run_log.txt'), 'w') as f:
            f.write(f"Model comparison completed successfully. Model selected: {best_model_name}\n")
            
        with open(os.path.join(OUTPUT_MODEL_DIR, 'run_log.txt'), 'w') as f:
            f.write(f"Best model ({best_model_name}) serialized correctly with test/val metrics.\n")
            
        with open(os.path.join(OUTPUT_COMPARE_DIR, 'run_metadata.json'), 'w') as f:
            json.dump({"task": "model_compare", "status": "success", "seed": 42}, f)
            
        with open(os.path.join(OUTPUT_MODEL_DIR, 'run_metadata.json'), 'w') as f:
            json.dump({"task": "best_model_save", "status": "success", "best_model": best_model_name}, f)
            
        print("Model comparison and evaluation script finished successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
