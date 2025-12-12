import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


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
        df['num_services'] = df[self.SERVICE_COLS].apply(
            lambda r: (r == 'Yes').sum(), axis=1
        )
        return df


class ChurnPreprocessor:
    def __init__(self, X: pd.DataFrame):
        self.numeric_cols = X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

    def build(self) -> ColumnTransformer:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        return ColumnTransformer([
            ('num', numeric_transformer, self.numeric_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])


class ChurnModelTrainer:
    def __init__(self, preprocessor: ColumnTransformer):
        self.pipeline = Pipeline([
            ('preproc', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', XGBClassifier(
                eval_metric='auc',
                verbosity=0,
                random_state=42
            ))
        ])

    def param_space(self) -> dict:
        return {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__subsample': [0.6, 0.8, 1.0],
            'clf__colsample_bytree': [0.6, 0.8, 1.0]
        }

    def train(self, X_train, y_train) -> RandomizedSearchCV:
        search = RandomizedSearchCV(
            self.pipeline,
            param_distributions=self.param_space(),
            n_iter=10,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            error_score='raise'
        )
        search.fit(X_train, y_train)
        return search

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, y_pred, digits=4))
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        return y_pred, y_proba

    def save(self, model, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)


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

        y = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop('Churn', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        preprocessor = ChurnPreprocessor(X_train).build()
        trainer = ChurnModelTrainer(preprocessor)

        search = trainer.train(X_train, y_train)
        best_model = search.best_estimator_

        y_pred, y_proba = trainer.evaluate(best_model, X_test, y_test)

        trainer.save(best_model, "results/churn_model_imputed.joblib")

        test_df = X_test.reset_index(drop=True)
        test_df['actual_churn'] = y_test.reset_index(drop=True)
        test_df['predicted_churn'] = y_pred
        test_df['predicted_proba'] = y_proba

        test_df.to_csv(
            os.path.join(self.output_dir, "test_set_results.csv"),
            index=False
        )


if __name__ == "__main__":
    ChurnPipelineApp("files/dataset.csv").run()
