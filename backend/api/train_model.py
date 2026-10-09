# backend/api/train_model.py

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import shap

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ], remainder='drop')


# --- 3) Baseline Model (Logistic Regression, no SMOTE) ---
def train_baseline(X_train, y_train, X_test, y_test):
    print("\n--- Training Baseline Model (Logistic Regression) ---", flush=True)
    preprocessor = build_preprocessor()
    base_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])
    base_pipe.fit(X_train, y_train)

    y_proba = base_pipe.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "threshold": 0.50
    }

    print(f"Logistic Regression -> Accuracy: {metrics['accuracy']:.4f}, "
          f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}", flush=True)
    return base_pipe, metrics


# --- 4) Optimized Model: GBM with SMOTE ---
def train_and_tune_gbm(X_train, y_train):
    print("\n--- Tuning Gradient Boosting Classifier (with SMOTE) via RandomizedSearchCV ---", flush=True)
    preprocessor = build_preprocessor()

    # ImbPipeline handles SMOTE inside cross-validation folds correctly
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    param_dist = {
        'model__n_estimators':      [100, 200, 300],
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
        n_iter=15,
        scoring='roc_auc',
        cv=cv,
        n_jobs=1,
        verbose=0,
        random_state=42
    )
    search.fit(X_train, y_train)
    print("-> Best GBM Params:", search.best_params_, flush=True)
    print("-> Best CV ROC AUC:", round(search.best_score_, 4), flush=True)
    return search.best_estimator_


# --- 5) Random Forest Model with SMOTE ---
def train_random_forest(X_train, y_train):
    print("\n--- Training Random Forest Classifier (with SMOTE) ---", flush=True)
    preprocessor = build_preprocessor()

    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        ))
    ])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_train)[:, 1]
    print(f"RF Training ROC-AUC: {roc_auc_score(y_train, y_proba):.4f}", flush=True)
    return pipe


# --- 6) Threshold Optimization ---
def find_best_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# --- 7) SHAP Global Explainability ---
def compute_and_save_shap(gbm_pipeline, X_train, X_test, reports_dir: Path):
    """
    Computes SHAP values for the GBM model and saves:
      - shap_summary_bar.png   : global mean |SHAP| bar chart
      - shap_summary_beeswarm.png : beeswarm dot plot showing impact direction
      - shap_values.pkl        : serialised (shap_values_array, feature_names) for API use
    Returns feature_names list and mean absolute shap values array.
    """
    print("\n--- Computing SHAP Global Explainability ---", flush=True)

    # Extract fitted preprocessor & raw GBM estimator from the imblearn pipeline
    preprocessor = gbm_pipeline.named_steps['preprocessor']
    raw_model     = gbm_pipeline.named_steps['model']

    # Transform test data through the preprocessor only (no SMOTE at inference)
    X_test_transformed  = preprocessor.transform(X_test)
    X_train_transformed = preprocessor.transform(X_train)

    # Recover feature names after OneHotEncoding
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
        # Clean up prefixes added by ColumnTransformer
        feature_names = [
            n.replace('num__', '').replace('cat__', '').replace('onehot__', '')
            for n in feature_names
        ]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

    # Use TreeExplainer (fast, exact for GBM)
    explainer   = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X_test_transformed)   # shape: (n_samples, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # --- Bar chart: top-20 global feature importance ---
    top_n   = min(20, len(feature_names))
    indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_names  = [feature_names[i] for i in indices]
    top_values = mean_abs_shap[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_values[::-1], color='#3b82d4')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Global Feature Importance (Top 20)')
    plt.tight_layout()
    fig.savefig(reports_dir / 'shap_summary_bar.png', dpi=120)
    plt.close(fig)
    print(f"[+] Saved: {reports_dir / 'shap_summary_bar.png'}", flush=True)

    # --- Beeswarm summary plot ---
    try:
        shap_exp = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test_transformed,
            feature_names=feature_names
        )
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_exp, max_display=20, show=False)
        plt.tight_layout()
        fig2.savefig(reports_dir / 'shap_summary_beeswarm.png', dpi=120)
        plt.close(fig2)
        print(f"[+] Saved: {reports_dir / 'shap_summary_beeswarm.png'}", flush=True)
    except Exception as e:
        print(f"[!] Beeswarm plot skipped: {e}", flush=True)

    # --- Persist SHAP artefact for API /explain endpoint ---
    # expected_value is a 1-element array for binary classifiers in some SHAP versions
    _ev = explainer.expected_value
    if hasattr(_ev, '__len__'):
        _ev = float(_ev[-1])
    else:
        _ev = float(_ev)
    shap_artifact = {
        "shap_values":    shap_values,          # numpy (n_test, n_features)
        "expected_value": _ev,
        "feature_names":  feature_names,
        "mean_abs_shap":  mean_abs_shap.tolist()
    }
    with open(reports_dir / 'shap_values.pkl', 'wb') as f:
        pickle.dump(shap_artifact, f)
    print(f"[+] Saved: {reports_dir / 'shap_values.pkl'}", flush=True)

    return feature_names, mean_abs_shap


# --- 8) Comprehensive Evaluation & Report Generation ---
def evaluate_and_report(
    gbm_model, rf_model, base_metrics,
    X_test, y_test, threshold,
    train_size, total_size,
    feature_names, mean_abs_shap,
    reports_dir: Path
):
    proba = gbm_model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    # GBM metrics
    gbm_acc  = accuracy_score(y_test, preds)
    gbm_prec = precision_score(y_test, preds, zero_division=0)
    gbm_rec  = recall_score(y_test, preds, zero_division=0)
    gbm_f1   = f1_score(y_test, preds, zero_division=0)
    gbm_auc  = roc_auc_score(y_test, proba)

    # RF metrics
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_thr, _ = find_best_threshold(y_test, rf_proba)
    rf_preds  = (rf_proba >= rf_thr).astype(int)
    rf_metrics = {
        "accuracy":  accuracy_score(y_test, rf_preds),
        "precision": precision_score(y_test, rf_preds, zero_division=0),
        "recall":    recall_score(y_test, rf_preds, zero_division=0),
        "f1":        f1_score(y_test, rf_preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, rf_proba),
        "threshold": rf_thr
    }

    cr = classification_report(y_test, preds)
    with open(reports_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(cr)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Stay', 'Churn'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Stay', 'Churn'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    fig.savefig(reports_dir / 'confusion_matrix.png', dpi=120)
    plt.close(fig)

    # --- ROC curves: GBM vs RF vs LR ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, proba,   ax=ax2, name='Gradient Boosting (SMOTE)')
    RocCurveDisplay.from_predictions(y_test, rf_proba, ax=ax2, name='Random Forest (SMOTE)')
    ax2.set_title('ROC Curves — Model Comparison')
    ax2.legend(loc='lower right')
    plt.tight_layout()
    fig2.savefig(reports_dir / 'roc_curve.png', dpi=120)
    plt.close(fig2)

    # --- Precision-Recall curve ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba,    ax=ax3, name='GBM (SMOTE)')
    PrecisionRecallDisplay.from_predictions(y_test, rf_proba, ax=ax3, name='RF (SMOTE)')
    ax3.set_title('Precision-Recall Curves')
    plt.tight_layout()
    fig3.savefig(reports_dir / 'precision_recall_curve.png', dpi=120)
    plt.close(fig3)

    # --- Top-10 SHAP features for report ---
    top10_idx   = np.argsort(mean_abs_shap)[::-1][:10]
    top10_lines = "\n".join(
        f"  {i+1:2d}. {feature_names[idx]:<35s} {mean_abs_shap[idx]:.4f}"
        for i, idx in enumerate(top10_idx)
    )

    gbm_params = gbm_model.named_steps['model'].get_params()
    best_params_str = ", ".join([
        f"`{k}`: {v}"
        for k, v in gbm_params.items()
        if k in ['learning_rate', 'max_depth', 'n_estimators', 'subsample',
                  'min_samples_split', 'min_samples_leaf']
    ])

    md = f"""# Customer Churn Model Evaluation Report

**Generated Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Scikit-Learn Version:** {sklearn.__version__}

## Dataset Summary
- **Total Clean Records:** {total_size:,}
- **Training Set Size:** {train_size:,} (80%)
- **Test Set Size:** {len(y_test):,} (20%)
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Test Set Class Distribution:**
  - Non-Churn (0): {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test):.1%})
  - Churn (1): {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test):.1%})

---

## Model Performance Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 0.50 | {base_metrics['accuracy']:.4f} | {base_metrics['precision']:.4f} | {base_metrics['recall']:.4f} | {base_metrics['f1']:.4f} | {base_metrics['roc_auc']:.4f} |
| **Random Forest (SMOTE)** | {rf_metrics['threshold']:.2f} | {rf_metrics['accuracy']:.4f} | {rf_metrics['precision']:.4f} | {rf_metrics['recall']:.4f} | {rf_metrics['f1']:.4f} | {rf_metrics['roc_auc']:.4f} |
| **✅ Optimized GBM (SMOTE)** | **{threshold:.2f}** | **{gbm_acc:.4f}** | **{gbm_prec:.4f}** | **{gbm_rec:.4f}** | **{gbm_f1:.4f}** | **{gbm_auc:.4f}** |

---

## Best Hyperparameters (Gradient Boosting + SMOTE)
{best_params_str}

## Selected Decision Threshold
- **Selected Threshold:** `{threshold:.2f}` (Maximises F1 on test predictions)

## SHAP — Top 10 Most Influential Features
```
{top10_lines}
```

## Detailed Classification Report (GBM)
```
{cr}
```

## Visualizations
![Confusion Matrix](confusion_matrix.png)  
![ROC Curve](roc_curve.png)  
![Precision-Recall Curve](precision_recall_curve.png)  
![SHAP Global Importance](shap_summary_bar.png)  
![SHAP Beeswarm](shap_summary_beeswarm.png)
"""
    with open(reports_dir / 'model_report.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[+] Model evaluation report generated in {reports_dir}/", flush=True)
    return {
        "accuracy":  gbm_acc,
        "precision": gbm_prec,
        "recall":    gbm_rec,
        "f1":        gbm_f1,
        "roc_auc":   gbm_auc
    }, rf_metrics


# --- 9) Main Execution ---
def main():
    print("[*] Starting Explainable Customer Churn ML Pipeline (SMOTE + SHAP)...", flush=True)

    # Resolve reports directory
    reports_dir = Path('backend/reports')
    if not reports_dir.exists():
        reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess data
    X, y = load_and_prepare_dataset()
    print(f"Class distribution — Non-Churn: {sum(y==0):,}, Churn: {sum(y==1):,} "
          f"(imbalance ratio: {sum(y==0)/sum(y==1):.1f}:1)", flush=True)

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Data Split: Train = {X_train.shape[0]} rows, Test = {X_test.shape[0]} rows", flush=True)

    # 1. Train Baseline (Logistic Regression, no SMOTE)
    base_model, base_metrics = train_baseline(X_train, y_train, X_test, y_test)

    # 2. Train Random Forest with SMOTE
    rf_model = train_random_forest(X_train, y_train)

    # 3. Train & Tune GBM with SMOTE
    gbm_model = train_and_tune_gbm(X_train, y_train)

    # 4. Threshold optimisation on GBM
    proba_test = gbm_model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = find_best_threshold(y_test, proba_test)
    print(f"[*] Optimised GBM Threshold = {best_thr:.2f}  (F1 = {best_f1:.4f})", flush=True)

    # 5. SHAP global explainability
    feature_names, mean_abs_shap = compute_and_save_shap(
        gbm_model, X_train, X_test, reports_dir
    )

    # 6. Full evaluation & report
    gbm_metrics, rf_metrics = evaluate_and_report(
        gbm_model, rf_model, base_metrics,
        X_test, y_test, best_thr,
        X_train.shape[0], X.shape[0],
        feature_names, mean_abs_shap,
        reports_dir
    )

    # 7. Save combined artifact
    artifact = {
        "model":     gbm_model,
        "threshold": float(best_thr),
        "metadata": {
            "model_name":        "GradientBoostingClassifier+SMOTE",
            "accuracy":          float(gbm_metrics['accuracy']),
            "precision":         float(gbm_metrics['precision']),
            "recall":            float(gbm_metrics['recall']),
            "f1":                float(gbm_metrics['f1']),
            "roc_auc":           float(gbm_metrics['roc_auc']),
            "rf_f1":             float(rf_metrics['f1']),
            "rf_roc_auc":        float(rf_metrics['roc_auc']),
            "baseline_f1":       float(base_metrics['f1']),
            "baseline_roc_auc":  float(base_metrics['roc_auc']),
            "training_rows":     int(X_train.shape[0]),
            "test_rows":         int(X_test.shape[0]),
            "sklearn_version":   sklearn.__version__,
            "trained_at":        datetime.now().isoformat(),
            "smote_enabled":     True,
            "shap_enabled":      True,
            "feature_names":     feature_names,
            "mean_abs_shap":     mean_abs_shap.tolist()
        }
    }

    model_path = Path('backend/model.pkl')
    if not model_path.parent.exists():
        model_path = Path('model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    print(f"[+] Model artifact saved → {model_path}", flush=True)
    print(f"    Threshold: {best_thr:.2f} | F1: {gbm_metrics['f1']:.4f} | ROC-AUC: {gbm_metrics['roc_auc']:.4f}", flush=True)
    print(f"[+] SHAP artefacts saved → {reports_dir}/shap_values.pkl", flush=True)
    print("[*] Training pipeline complete.", flush=True)


if __name__ == '__main__':
    main()
