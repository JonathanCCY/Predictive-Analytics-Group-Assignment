import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_pipeline():
    try:
        # Paths
        OUTPUT_SHARED_DIR = 'outputs/shared'
        OUTPUT_DEBUG_DIR = 'outputs/debug'
        os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
        
        print("Loading data...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')

        # Combine datasets to create a custom split later
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # FIX: Drop the unnamed index and 'id' columns
        drop_cols = [c for c in df.columns[:5] if 'Unnamed' in c or c.lower() == 'id']
        if 'id' in df.columns and 'id' not in drop_cols:
            drop_cols.append('id')
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        print("Preprocessing data...")
        # FIX: Encode target using defined integer mappings correctly
        df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
        
        X = df.drop('satisfaction', axis=1)
        y = df['satisfaction']
        
        # Automatically identify feature types based on valid DataFrame
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # FIX: Re-use canonical split from Step 2 instead of an unseeded random split
        manifest_path = os.path.join(OUTPUT_SHARED_DIR, 'split_manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        # Stratified 70/15/15 using random_state=42 exactly as manifest
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # FIX: Adding sparse_output=False matching specific requirement 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # FIX: Create a full pipeline to strictly avert target leakage (fitting prep on all data)
        # FIX: Maintain LogisticRegression defaults with seed and specific solver
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
        ])
        
        print("Training model...")
        # FIX: Train using y_train, original code mistakenly passed y_temp
        model.fit(X_train, y_train)
        
        def evaluate_and_save(model_obj, X_eval, y_eval, split_name):
            probs = model_obj.predict_proba(X_eval)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            metrics = {
                "ROC-AUC": roc_auc_score(y_eval, probs),
                "PR-AUC": average_precision_score(y_eval, probs),
                "Accuracy": accuracy_score(y_eval, preds),
                "Precision": precision_score(y_eval, preds, zero_division=0),
                "Recall": recall_score(y_eval, preds, zero_division=0),
                "F1-score": f1_score(y_eval, preds, zero_division=0)
            }
            
            with open(os.path.join(OUTPUT_DEBUG_DIR, f'metrics_{split_name}.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
                
            pd.DataFrame({"probability": probs, "prediction": preds}).to_csv(
                os.path.join(OUTPUT_DEBUG_DIR, f'{split_name}_predictions.csv'), index=False)
                
            pd.DataFrame(confusion_matrix(y_eval, preds)).to_csv(
                os.path.join(OUTPUT_DEBUG_DIR, f'confusion_matrix_{split_name}.csv'), index=False)
                
            return metrics

        print("Evaluating model...")
        val_metrics = evaluate_and_save(model, X_val, y_val, "validation")
        test_metrics = evaluate_and_save(model, X_test, y_test, "test")
        
        print(f"Validation Accuracy: {val_metrics['Accuracy']:.4f}")
        
        with open(os.path.join(OUTPUT_DEBUG_DIR, 'feature_manifest.json'), 'w') as f:
            json.dump({"numeric_features": numeric_features, "categorical_features": categorical_features}, f, indent=4)
            
        print("Saving model...")
        # FIX: Save model using joblib instead of pickle with a text wrapper
        joblib.dump(model, os.path.join(OUTPUT_DEBUG_DIR, 'model.joblib'))
        
        with open(os.path.join(OUTPUT_DEBUG_DIR, 'run_log.txt'), 'w') as f:
            f.write("Debug pipeline executed successfully. Model saved and metrics exported.\n")
            
        with open(os.path.join(OUTPUT_DEBUG_DIR, 'run_metadata.json'), 'w') as f:
            json.dump({"task": "debug_pipeline", "status": "success"}, f)
            
        print("Pipeline finished successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        
if __name__ == "__main__":
    run_pipeline()
