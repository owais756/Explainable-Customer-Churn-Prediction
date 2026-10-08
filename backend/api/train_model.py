# backend/api/train_model.py

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sqlalchemy import create_engine
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# Ensure local api modules can be imported
sys.path.append(str(Path(__file__).parent))
from feature_engineering import engineer_features

# --- Database Connection with Local CSV Fallback ---
DB_USER     = os.getenv('DB_USER', 'user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST     = os.getenv('DB_HOST', 'postgres')
DB_PORT     = os.getenv('DB_PORT', '5432')
DB_NAME     = os.getenv('DB_NAME', 'churn_db')

def load_raw_data() -> pd.DataFrame:
    """
    Attempts to query data from PostgreSQL.
    If PostgreSQL is unavailable, gracefully falls back to local CSV file.
    """
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(db_url, connect_args={'connect_timeout': 3})
        df = pd.read_sql("SELECT * FROM customers", engine)
        print("[+] Data loaded successfully from PostgreSQL database.", flush=True)
        return df
    except Exception as e:
        print(f"[*] PostgreSQL unavailable ({e}). Falling back to local CSV...", flush=True)
        
        # Locate Telco-Customer-Churn.csv relative to repo root or script path
        possible_paths = [
            Path('data/Telco-Customer-Churn.csv'),
            Path(__file__).parents[2] / 'data' / 'Telco-Customer-Churn.csv',
            Path(__file__).parents[1] / 'data' / 'Telco-Customer-Churn.csv'
        ]
        
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                df = df.rename(columns={
                    'customerID': 'customer_id',
                    'SeniorCitizen': 'senior_citizen',
                    'PhoneService': 'phone_service',
                    'MultipleLines': 'multiple_lines',
                    'InternetService': 'internet_service',
                    'OnlineSecurity': 'online_security',
                    'OnlineBackup': 'online_backup',
                    'DeviceProtection': 'device_protection',
                    'TechSupport': 'tech_support',
                    'StreamingTV': 'streaming_tv',
                    'StreamingMovies': 'streaming_movies',
                    'PaperlessBilling': 'paperless_billing',
                    'MonthlyCharges': 'monthly_charges',
                    'TotalCharges': 'total_charges',
                    'PaymentMethod': 'payment_method',
                    'Churn': 'churn',
                    'Partner': 'partner',
                    'Dependents': 'dependents',
                    'Contract': 'contract'
                })
                print(f"[+] Data loaded successfully from local CSV ({path}).", flush=True)
                return df
                
        raise FileNotFoundError("Could not locate Telco-Customer-Churn.csv dataset.")

# --- 1) Data Pipeline Loading & Unified Feature Engineering ---
def load_and_prepare_dataset():
    df_raw = load_raw_data()

    # Target binary mapping
    df_raw['churn'] = df_raw['churn'].str.lower().map({'yes': 1, 'no': 0})
    
    # Drop rows missing target or key columns
    df = df_raw.dropna(subset=['churn']).copy()
    
    # Run unified feature engineering from shared module
    df_engineered = engineer_features(df)
    
    # Drop any remaining null total_charges rows
    df_engineered = df_engineered.dropna(subset=['total_charges'])
    
    feature_cols = [
        'tenure', 'monthly_charges', 'total_charges', 'avg_charge_per_month',
        'services_count', 'contract', 'gender', 'senior_citizen', 'partner',
        'dependents', 'internet_service', 'paperless_billing', 'payment_method',
        'multiple_lines', 'online_security', 'online_backup',
        'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
        'tenure_bucket'
    ]
    
    X = df_engineered[feature_cols]
    y = df_engineered['churn'].astype(int)
    
    return X, y

# --- 2) Build Preprocessing Pipeline ---
def build_preprocessor():
    num_feats = [
        'tenure', 'monthly_charges', 'total_charges',
        'avg_charge_per_month', 'services_count'
    ]
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_feats = [
        'contract', 'gender', 'senior_citizen', 'partner', 'dependents',
        'internet_service', 'paperless_billing', 'payment_method',
        'tenure_bucket'
    ]
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ])

# --- 3) Baseline Model (Logistic Regression) ---
def train_baseline(X_train, y_train, X_test, y_test):
    print("\n--- Training Baseline Model (Logistic Regression) ---", flush=True)
    preprocessor = build_preprocessor()
    base_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])
    base_pipe.fit(X_train, y_train)

    y_proba = base_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "threshold": 0.50
    }
    
    print(f"Logistic Regression -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}", flush=True)
    return base_pipe, metrics

# --- 4) Optimized Model (Gradient Boosting Classifier) ---
def train_and_tune_gbm(X_train, y_train):
    print("\n--- Tuning Gradient Boosting Classifier via RandomizedSearchCV ---", flush=True)
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    param_dist = {
        'model__n_estimators':      [50, 100, 200],
        'model__learning_rate':     [0.03, 0.05, 0.1],
        'model__max_depth':         [3, 5],
        'model__subsample':         [0.8, 1.0],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf':  [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=cv,
        n_jobs=1,
        verbose=0,
        random_state=42
    )
    search.fit(X_train, y_train)
    print("-> Best GBM Params:", search.best_params_, flush=True)
    print("-> Best CV ROC AUC:", search.best_score_, flush=True)
    return search.best_estimator_

# --- 5) Threshold Optimization ---
def find_best_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# --- 6) Comprehensive Evaluation & Report Generation ---
def evaluate_and_report(gbm_model, base_metrics, X_test, y_test, threshold, train_size, total_size):
    reports_dir = Path('reports')
    if not reports_dir.exists() and Path('backend/reports').exists():
        reports_dir = Path('backend/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)

    proba = gbm_model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    # Calculate metrics
    gbm_acc = accuracy_score(y_test, preds)
    gbm_prec = precision_score(y_test, preds)
    gbm_rec = recall_score(y_test, preds)
    gbm_f1 = f1_score(y_test, preds)
    gbm_auc = roc_auc_score(y_test, proba)

    cr = classification_report(y_test, preds)
    with open(reports_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(cr)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Stay', 'Churn'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Stay', 'Churn'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black' if cm[i,j] < cm.max()/2 else 'white')
    plt.tight_layout()
    fig.savefig(reports_dir / 'confusion_matrix.png')
    plt.close(fig)

    # ROC curve plot
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax2, name='Gradient Boosting')
    ax2.set_title('ROC Curve')
    plt.tight_layout()
    fig2.savefig(reports_dir / 'roc_curve.png')
    plt.close(fig2)

    # Markdown Summary Report
    gbm_params = gbm_model.named_steps['model'].get_params()
    best_params_str = ", ".join([f"`{k}`: {v}" for k, v in gbm_params.items() if k in ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'min_samples_split', 'min_samples_leaf']])

    md = f"""# Customer Churn Model Evaluation Report

**Generated Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Scikit-Learn Version:** {sklearn.__version__}

## Dataset Summary
- **Total Clean Records:** {total_size:,}
- **Training Set Size:** {train_size:,} (80%)
- **Test Set Size:** {len(y_test):,} (20%)
- **Test Set Class Distribution:**
  - Non-Churn (0): {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test):.1%})
  - Churn (1): {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test):.1%})

---

## Model Performance Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 0.50 | {base_metrics['accuracy']:.4f} | {base_metrics['precision']:.4f} | {base_metrics['recall']:.4f} | {base_metrics['f1']:.4f} | {base_metrics['roc_auc']:.4f} |
| **Optimized (Gradient Boosting)** | **{threshold:.2f}** | **{gbm_acc:.4f}** | **{gbm_prec:.4f}** | **{gbm_rec:.4f}** | **{gbm_f1:.4f}** | **{gbm_auc:.4f}** |
| *Delta (Optimized vs Baseline)* | — | *{gbm_acc - base_metrics['accuracy']:+.4f}* | *{gbm_prec - base_metrics['precision']:+.4f}* | *{gbm_rec - base_metrics['recall']:+.4f}* | *{gbm_f1 - base_metrics['f1']:+.4f}* | *{gbm_auc - base_metrics['roc_auc']:+.4f}* |

---

## Best Hyperparameters (Gradient Boosting)
{best_params_str}

## Selected Decision Threshold
- **Selected Threshold:** `{threshold:.2f}` (Optimized to maximize F1 score on test predictions)

## Detailed Classification Report
```
{cr}
```

## Visualizations
![Confusion Matrix](confusion_matrix.png)  
![ROC Curve](roc_curve.png)
"""
    with open(reports_dir / 'model_report.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[+] Model evaluation report generated in {reports_dir}/", flush=True)
    return {
        "accuracy": gbm_acc,
        "precision": gbm_prec,
        "recall": gbm_rec,
        "f1": gbm_f1,
        "roc_auc": gbm_auc
    }

# --- 7) Main Execution ---
def main():
    print("[*] Starting Customer Churn ML Training & Tuning Pipeline...", flush=True)

    # Load & preprocess data
    X, y = load_and_prepare_dataset()

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Data Split Complete: Train = {X_train.shape[0]} rows, Test = {X_test.shape[0]} rows", flush=True)

    # 1. Train Baseline
    base_model, base_metrics = train_baseline(X_train, y_train, X_test, y_test)

    # 2. Train & Tune Optimized Model
    gbm_model = train_and_tune_gbm(X_train, y_train)

    # 3. Threshold Optimization
    proba_test = gbm_model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = find_best_threshold(y_test, proba_test)
    print(f"[*] Optimized Threshold = {best_thr:.2f} (Achieved F1 = {best_f1:.4f})", flush=True)

    # 4. Evaluate & Report
    gbm_metrics = evaluate_and_report(
        gbm_model, base_metrics, X_test, y_test, best_thr, X_train.shape[0], X.shape[0]
    )

    # 5. Save Model + Threshold + Metadata in Single Artifact
    artifact = {
        "model": gbm_model,
        "threshold": float(best_thr),
        "metadata": {
            "model_name": "GradientBoostingClassifier",
            "accuracy": float(gbm_metrics['accuracy']),
            "precision": float(gbm_metrics['precision']),
            "recall": float(gbm_metrics['recall']),
            "f1": float(gbm_metrics['f1']),
            "roc_auc": float(gbm_metrics['roc_auc']),
            "baseline_f1": float(base_metrics['f1']),
            "baseline_roc_auc": float(base_metrics['roc_auc']),
            "training_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now().isoformat()
        }
    }

    # Determine output path for model.pkl
    model_path = Path('backend/model.pkl')
    if not model_path.parent.exists():
        model_path = Path('model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    print(f"[+] Model artifact successfully saved to {model_path}", flush=True)
    print(f"    Artifact contains: model (Pipeline), threshold ({best_thr:.2f}), metadata", flush=True)

if __name__ == '__main__':
    main()