#!/usr/bin/env python3
"""
model.py

Loads the Telco churn dataset, performs feature engineering,
builds a preprocessing (+ imputation) → SMOTE → XGBoost pipeline
with randomized search hyperparameter tuning, evaluates performance,
and saves the final model to disk.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score

# ← CHANGE: import Pipeline from imblearn, not sklearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# 1. Load data
df = pd.read_csv('files/dataset.csv')
df = df.drop('customerID', axis=1, errors='ignore')

# === Save test‐set detailed outputs ===
out_dir = 'model_output'
os.makedirs(out_dir, exist_ok=True)

# 2. Feature engineering
df['tenure_group'] = pd.cut(
    df.tenure,
    bins=[0,12,24,48,60,np.inf],
    labels=['0-12','13-24','25-48','49-60','61+']
)
services = [
    'OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies'
]
df['num_services'] = df[services].apply(lambda row: (row=='Yes').sum(), axis=1)

# 3. Define target and features
y = df['Churn'].map({'Yes':1, 'No':0})
X = df.drop('Churn', axis=1)

# 4. Column lists
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

# 5. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,     numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
])

# 6. Full pipeline: preproc → SMOTE → classifier
pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('smote',   SMOTE(random_state=42)),
    ('clf',     XGBClassifier(
        eval_metric='auc',
        verbosity=0,
        random_state=42
    ))
])

# 7. Hyperparameter search space
param_dist = {
    'clf__n_estimators':      [100, 200, 300],
    'clf__max_depth':         [3, 5, 7],
    'clf__learning_rate':     [0.01, 0.1, 0.2],
    'clf__subsample':         [0.6, 0.8, 1.0],
    'clf__colsample_bytree':  [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    error_score='raise'
)

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 9. Run search
print("Starting hyperparameter search...")
search.fit(X_train, y_train)

print(f"\nBest parameters:\n{search.best_params_}\n")
best_model = search.best_estimator_

# 10. Evaluate on test set
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

print("Test set performance:")
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 11. Save the trained pipeline
model_path = 'results/churn_model_imputed.joblib'
joblib.dump(best_model, model_path)
print(f"\nSaved trained model to: {model_path}")

# Reconstruct a DataFrame with raw X_test, true & predicted values
test_df = X_test.copy().reset_index(drop=True)
test_df['actual_churn']    = y_test.reset_index(drop=True)
test_df['predicted_churn'] = y_pred
test_df['predicted_proba'] = y_proba

# Write to CSV
test_df.to_csv(os.path.join(out_dir, 'test_set_results.csv'), index=False)
print(f"Detailed test‐set results saved to {out_dir}/test_set_results.csv")
