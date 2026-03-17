import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_pipeline():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # Combine datasets to create a custom split later
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    print("Preprocessing data...")
    # Encode target variable
    df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
    
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    # Automatically identify feature types
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing to the dataset
    X_processed = preprocessor.fit_transform(X)
    
    print("Splitting data into 70/15/15...")
    # Split the data into train, val, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
    
    print("Training model...")
    model = LogisticRegression(max_iter=500)
    
    # Fit the model
    model.fit(X_train, y_temp)
    
    print("Evaluating model...")
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("Test Set Performance:")
    print(classification_report(y_val, val_preds))
    
    print("Saving model...")
    with open('outputs/baseline_model.pkl', 'w') as f:
        pickle.dump(model, f)
        
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    run_pipeline()
