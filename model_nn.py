#!/usr/bin/env python3
"""
model_nn.py

A feed‑forward NN for churn prediction with Keras:
- Imputes & scales numerics
- One‑hot encodes categoricals
- Trains with early stopping on val AUC
- Evaluates & saves model + preprocessor
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

# 1) Load & feature-engineer
df = (
    pd.read_csv('files/dataset.csv')
      .drop('customerID', axis=1, errors='ignore')
)
df['tenure_group'] = pd.cut(
    df.tenure,
    bins=[0,12,24,48,60,np.inf],
    labels=['0‑12','13‑24','25‑48','49‑60','61+']
)
services = ['OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies']
df['num_services'] = df[services].eq('Yes').sum(axis=1)

# 2) Define X, y
y = df['Churn'].map({'Yes':1,'No':0}).values
X = df.drop('Churn', axis=1)

# 3) Split into train/val/test (60/20/20)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42
)

# 4) Build preprocessing pipeline
numeric_cols     = X.select_dtypes(['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(['object','category']).columns.tolist()

num_pipe = SKPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
cat_pipe = SKPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe,     numeric_cols),
    ('cat', cat_pipe,     categorical_cols)
])

# fit & transform to NumPy arrays
X_train_np = preprocessor.fit_transform(X_train)
X_val_np   = preprocessor.transform(X_val)
X_test_np  = preprocessor.transform(X_test)

input_dim = X_train_np.shape[1]

# 5) Build the model
def build_model():
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(128, activation='relu')(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

model = build_model()
model.summary()

# 6) Train with early stopping on validation AUC
es = callbacks.EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_np, y_train,
    validation_data=(X_val_np, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[es],
    verbose=2
)

# 7) Evaluate on test set
y_proba = model.predict(X_test_np).ravel()
y_pred  = (y_proba >= 0.5).astype(int)

print("\n=== Test Set Metrics ===")
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 8) Save model + preprocessor
os.makedirs('model_output', exist_ok=True)
model.save('model_output/churn_nn.h5')
joblib.dump(preprocessor, 'model_output/preprocessor_nn.joblib')

print("\nSaved Keras model to model_output/churn_nn.h5")
print("Saved preprocessor to model_output/preprocessor_nn.joblib")
