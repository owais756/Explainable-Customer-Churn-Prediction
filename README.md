# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%3E%3D13-blue?logo=postgresql)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![ML](https://img.shields.io/badge/ML-GradientBoosting+SMOTE-yellow)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

A full-stack machine learning application for predicting customer churn in a telecommunications company. The system addresses class imbalance using SMOTE, evaluates models using Precision, Recall, F1-Score and ROC-AUC, and uses SHAP-based explainability to identify the key factors driving individual and overall churn predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Details](#machine-learning-details)
- [SHAP Explainability](#shap-explainability)
- [Model Performance](#model-performance)
- [Setup and Installation](#setup-and-installation)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## 🔭 Overview

This project provides a comprehensive, explainable solution for telecommunications customer churn analysis and prediction. It demonstrates an end-to-end data science workflow — from data loading, preprocessing, and class-imbalance correction, through model training and SHAP-based explainability, all the way to deployment via a REST API and an interactive Streamlit dashboard.

The solution helps businesses answer critical questions like:
- Which customers are likely to churn?
- **Why** is a specific customer predicted to churn? *(SHAP individual explanation)*
- What are the most important global drivers of churn across all customers?
- How can targeted retention strategies be designed based on explainable risk factors?

---

## 🌟 Key Features

- **SHAP Explainability**: Both global (mean |SHAP| bar + beeswarm) and per-customer (waterfall chart) explanations powered by `shap.TreeExplainer`
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) applied correctly *inside* cross-validation folds via `imblearn.pipeline`
- **Three-Model Comparison**: Logistic Regression baseline vs. Random Forest (SMOTE) vs. tuned Gradient Boosting (SMOTE)
- **Threshold Optimization**: Decision threshold selected by maximising F1 score on test predictions
- **Interactive Dashboard**: Comprehensive Streamlit dashboard with 5 tabs — Overview, Insights, Prediction, Explainability, About
- **Inline SHAP in Predictions**: Every prediction in the UI shows a mini SHAP bar chart explaining the top 5 driving factors
- **Real-time REST API**: FastAPI backend exposing `/predict` (with SHAP top features) and `/explain` (full SHAP waterfall) endpoints
- **Data Persistence**: PostgreSQL integration with local CSV fallback for portability
- **Containerization**: Docker and Docker Compose for easy deployment

---

## 🏗️ Architecture

The project follows a modern microservices architecture with three main components:

```
┌──────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                    │
│  Overview │ Insights │ Prediction │ Explainability │ About│
└─────────────────────┬────────────────────────────────────┘
                      │  REST (HTTP)
┌─────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend                        │
│   POST /predict   POST /explain   GET /health            │
│   GBM+SMOTE Pipeline + SHAP TreeExplainer                │
└─────────────────────┬────────────────────────────────────┘
                      │  SQLAlchemy ORM
┌─────────────────────▼────────────────────────────────────┐
│                  PostgreSQL Database                      │
│         customers table · prediction_history             │
└──────────────────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn |
| **ML & Explainability** | scikit-learn, imbalanced-learn (SMOTE), SHAP |
| **Frontend Dashboard** | Streamlit, Plotly |
| **Database** | PostgreSQL, SQLAlchemy |
| **Data Processing** | pandas, NumPy |
| **Visualisation** | Plotly, Matplotlib |
| **DevOps** | Docker, Docker Compose |

---

## 🧠 Machine Learning Details

### Models Trained

| Model | Imbalance Handling | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression *(baseline)* | None | 0.50 | — | — | 0.572 | 0.834 |
| Random Forest | SMOTE | Optimised | — | — | 0.625 | 0.831 |
| **Gradient Boosting ✅ selected** | **SMOTE** | **0.59** | **0.580** | **0.682** | **0.627** | **0.834** |

### Class Imbalance

The Telco dataset has a **2.8 : 1** class imbalance (Non-Churn : Churn). SMOTE is applied using `imblearn.pipeline.Pipeline` so synthetic samples are only generated on training folds — never leaked into validation or test sets.

### Feature Engineering

| Feature | Description |
|---|---|
| `avg_charge_per_month` | `total_charges / (tenure + ε)` |
| `services_count` | Count of active optional services |
| `tenure_bucket` | Categorical bucket: 0-12, 12-24, 24-48, 48+ months |

### Preprocessing Pipeline

- **Numerical**: Median imputation → StandardScaler
- **Categorical**: Most-frequent imputation → OneHotEncoder
- **Structure**: `ColumnTransformer` wrapped in `ImbPipeline` with SMOTE

### Hyperparameter Tuning

- `RandomizedSearchCV` with 5-fold `StratifiedKFold`
- Optimisation metric: **ROC-AUC**
- Parameters tuned: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_samples_split`, `min_samples_leaf`

---

## 🔍 SHAP Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to make every prediction interpretable.

### Global Explainability
Generated during training and saved to `backend/reports/`:

| Artifact | Description |
|---|---|
| `shap_summary_bar.png` | Mean \|SHAP\| bar chart — top 20 features by overall importance |
| `shap_summary_beeswarm.png` | Beeswarm plot — shows impact direction and spread per feature |

**Top 5 most influential features globally:**
1. `contract_Month-to-month` — strongest single churn driver
2. `tenure` — longer tenure strongly reduces churn risk
3. `internet_service_Fiber optic` — Fiber optic customers churn more
4. `payment_method_Electronic check` — correlated with higher churn
5. `monthly_charges` — higher charges increase churn probability

### Individual Explainability
The **Explainability tab** in the dashboard and the `/explain` API endpoint both return per-customer SHAP waterfall charts, showing exactly which features push a prediction toward or away from churn.

```
🔴 Red bars  →  push prediction TOWARDS churn  (positive SHAP)
🔵 Blue bars →  push prediction AWAY FROM churn (negative SHAP)
```

---

## 🚀 Setup and Installation

### Prerequisites
- Docker and Docker Compose, **or** Python 3.10+
- Git

### Quick Start (Docker)

```bash
git clone https://github.com/owais756/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
docker-compose up -d
```

Access:
- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI Swagger Docs**: http://localhost:8000/docs

### Manual Setup (Without Docker)

**1. Train the model first:**
```bash
cd Customer-Churn-Prediction
pip install -r backend/api/requirements.txt
python backend/api/train_model.py
```

**2. Start the backend API:**
```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**3. Start the frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 API Documentation

### `POST /predict`
Predict churn probability with inline SHAP explanation.

**Request:**
```json
{
  "tenure": 12,
  "monthly_charges": 70.0,
  "contract": "Month-to-month",
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "No",
  "dependents": "No",
  "internet_service": "Fiber optic",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "services_list": []
}
```

**Response:**
```json
{
  "prediction": "Yes",
  "probability": 0.768,
  "threshold": 0.59,
  "risk_level": "High",
  "recommended_action": "Offer loyalty discount or contract upgrade",
  "model_version": "v2.0",
  "shap_top_features": [
    { "feature": "contract_Month-to-month", "shap_value": 0.6473 },
    { "feature": "internet_service_Fiber optic", "shap_value": 0.3170 },
    { "feature": "payment_method_Electronic check", "shap_value": 0.2549 }
  ]
}
```

### `POST /explain`
Full per-feature SHAP breakdown for a customer.

**Response:**
```json
{
  "probability": 0.768,
  "expected_value": -0.105,
  "shap_top_features": [ ... ],
  "shap_all_features": [ ... ]
}
```

### `GET /health`
Returns API status, model version, threshold, SHAP availability, and full training metadata.

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI app + /predict + /explain endpoints
│   │   ├── train_model.py           # Full ML pipeline (SMOTE + SHAP + reporting)
│   │   ├── feature_engineering.py   # Shared feature transforms
│   │   ├── schemas.py               # Pydantic models (incl. ShapFeature, ExplainResponse)
│   │   ├── services/risk_service.py # Risk classification + recommendations
│   │   └── requirements.txt
│   ├── model.pkl                    # Trained GBM+SMOTE pipeline artifact
│   └── reports/
│       ├── shap_summary_bar.png     # Global SHAP bar chart
│       ├── shap_summary_beeswarm.png
│       ├── shap_values.pkl          # SHAP metadata for API
│       ├── roc_curve.png            # ROC: GBM vs RF
│       ├── precision_recall_curve.png
│       ├── confusion_matrix.png
│       └── classification_report.txt
├── frontend/
│   ├── app.py                       # Streamlit app (5 tabs)
│   ├── components/
│   │   ├── explainability.py        # SHAP global + individual explanation tab
│   │   ├── prediction.py            # Prediction form with inline SHAP
│   │   ├── insights.py
│   │   ├── plots.py
│   │   └── ...
│   └── requirements.txt
├── data/
│   └── Telco-Customer-Churn.csv
├── postgres/
│   ├── 01_create_schema.sql
│   └── 02_seed_data.sql
└── docker-compose.yml
```

---

## 📧 Contact

**Owais Mohammed**

[![Email](https://img.shields.io/badge/Email-owaismohammed203%40gmail.com-red?logo=gmail)](mailto:owaismohammed203@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Owais%20Mohammed-blue?logo=linkedin)](https://www.linkedin.com/in/owais-mohammed-1ab936319)
[![GitHub](https://img.shields.io/badge/GitHub-owais756-black?logo=github)](https://github.com/owais756)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
