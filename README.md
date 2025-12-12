# Customer Churn Prediction – Telecommunication Company

This project focuses on **predicting customer churn** for a telecommunication company using multiple machine learning approaches, ranging from classical ML models to advanced pipelines and neural networks.

The repository is designed to demonstrate an **end‑to‑end churn prediction workflow**, including data exploration, feature engineering, model training, evaluation, and model persistence.

---

## 📁 Project Structure

```
churn_prediction_telecommunication_company/
│
├── files/
│   └── dataset.csv                # Raw churn dataset
│
├── exploratory_data_analysis.py   # Automated EDA & statistical analysis
│
├── model_classic.py               # Classical ML models & ensemble learning
├── model_improved.py              # Advanced ML pipeline (SMOTE + XGBoost)
├── model_nn.py                    # Neural network churn model
│
├── graph/                         # Model evaluation plots
├── graphs_eda/                    # EDA visualizations
│
├── result/                        # Metrics, feature importance, predictions
├── results/                       # Additional outputs
├── model_output/                  # Saved trained models
│
├── requirements.txt
└── README.md
```

---

## 📊 Exploratory Data Analysis (EDA)

**Script:** `exploratory_data_analysis.py`

This module performs a complete exploratory analysis of the churn dataset.

### Key analyses

* Dataset statistics (shape, types, missing values)
* Churn rate analysis across numerical bins
* Categorical churn distributions
* Pair plots and scatter plots
* Boxplots (churn vs numerical features)
* Correlation heatmaps
* Categorical correlation using **Cramér’s V**

### Outputs

* All plots saved to `graphs_eda/`
* Statistical summaries saved to `results_eda/`

### Run

```bash
python exploratory_data_analysis.py
```

---

## 🤖 Classical Machine Learning Models

**Script:** `model_classic.py`

This script benchmarks a wide range of classical machine learning models using cross‑validation.

### Models included

* Logistic Regression
* Ridge Classifier
* K‑Nearest Neighbors
* Naive Bayes
* Random Forest
* Gradient Boosting
* AdaBoost
* XGBoost
* Voting Classifier (ensemble)

### Features

* Label encoding & scaling
* Cross‑validated model comparison
* Hyperparameter tuning
* Feature importance extraction
* Confusion matrices
* Final dataset with predictions

### Outputs

* Model comparison table (`result/model_creation.csv`)
* Feature importance CSV files
* Confusion matrix plots
* Final prediction dataset

### Run

```bash
python model_classic.py
```

---

## 🚀 Improved ML Pipeline (Production‑Style)

**Script:** `model_improved.py`

This version introduces a **clean, modular, object‑oriented pipeline** suitable for production‑grade ML systems.

### Enhancements

* Explicit feature engineering:

  * Tenure grouping
  * Number of subscribed services
* Robust preprocessing:

  * Missing value imputation
  * Standard scaling
  * One‑hot encoding
* Class imbalance handling using **SMOTE**
* Hyperparameter optimization with **RandomizedSearchCV**
* Final estimator: **XGBoost**

### Evaluation

* Classification report
* ROC AUC score

### Saved artifacts

* Trained pipeline (`.joblib`)
* Test‑set predictions with probabilities

### Run

```bash
python model_improved.py
```

---

## 🧠 Neural Network Model

**Script:** `model_nn.py`

A deep learning approach for churn prediction using Keras.

### Model characteristics

* Fully connected neural network
* Dropout regularization
* Early stopping on validation AUC
* End‑to‑end preprocessing and modeling

### Architecture overview

* Dense layers with ReLU activation
* Sigmoid output layer
* Optimized for ROC AUC

### Outputs

* Trained neural network (`.h5`)
* Saved preprocessing pipeline (`.joblib`)

### Run

```bash
python model_nn.py
```

---

## 📈 Evaluation Metrics

All models are evaluated using:

* Accuracy
* Precision
* Recall
* F1‑score
* ROC AUC
* Confusion Matrix

---

## ⚙️ Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### Main dependencies

* pandas, numpy
* scikit‑learn
* xgboost
* imbalanced‑learn
* matplotlib, seaborn
* tensorflow / keras
* joblib

---

## 🎯 Use Cases

* Customer churn risk prediction
* Retention strategy optimization
* Feature importance analysis
* Model benchmarking on telecom data
* Demonstration of end‑to‑end ML pipelines

---

## 👤 Author

**Ghazal Askari**
Senior Applied Machine Learning Engineer
Specialized in production ML systems, NLP, and predictive analytics

---

## 📄 License

This project is intended for educational and research purposes.
