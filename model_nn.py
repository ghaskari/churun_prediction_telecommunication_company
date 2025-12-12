import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import classification_report, roc_auc_score

from tensorflow.keras import layers, callbacks, models


class ChurnFeatureEngineer:
    SERVICE_COLS = [
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 60, np.inf],
            labels=['0-12', '13-24', '25-48', '49-60', '61+']
        )
        df['num_services'] = df[self.SERVICE_COLS].eq('Yes').sum(axis=1)
        return df


class ChurnPreprocessor:
    def __init__(self, X: pd.DataFrame):
        self.numeric_cols = X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        self.preprocessor = ColumnTransformer([
            (
                'num',
                SKPipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]),
                self.numeric_cols
            ),
            (
                'cat',
                SKPipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(
                        drop='first',
                        sparse=False,
                        handle_unknown='ignore'
                    ))
                ]),
                self.categorical_cols
            )
        ])

    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        return self.preprocessor.transform(X)


class ChurnNeuralNetwork:
    def __init__(self, input_dim: int):
        self.model = self._build(input_dim)

    def _build(self, input_dim: int):
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(128, activation='relu')(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inp, out)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        es = callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            restore_best_weights=True
        )

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=[es],
            verbose=2
        )

    def predict(self, X):
        return self.model.predict(X).ravel()

    def save(self, path: str):
        self.model.save(path)


class ChurnPipelineApp:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.output_dir = "model_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df.drop('customerID', axis=1, errors='ignore')

    def run(self):
        df = self.load_data()
        df = ChurnFeatureEngineer().transform(df)

        y = df['Churn'].map({'Yes': 1, 'No': 0}).values
        X = df.drop('Churn', axis=1)

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42
        )

        preprocessor = ChurnPreprocessor(X_train)

        X_train_np = preprocessor.fit_transform(X_train)
        X_val_np = preprocessor.transform(X_val)
        X_test_np = preprocessor.transform(X_test)

        model = ChurnNeuralNetwork(X_train_np.shape[1])
        model.train(X_train_np, y_train, X_val_np, y_val)

        y_proba = model.predict(X_test_np)
        y_pred = (y_proba >= 0.5).astype(int)

        print(classification_report(y_test, y_pred, digits=4))
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

        model.save(os.path.join(self.output_dir, "churn_nn.h5"))
        joblib.dump(
            preprocessor.preprocessor,
            os.path.join(self.output_dir, "preprocessor_nn.joblib")
        )


if __name__ == "__main__":
    ChurnPipelineApp("files/dataset.csv").run()
