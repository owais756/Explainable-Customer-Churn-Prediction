# backend/api/main.py

import os
import sys
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Ensure local api modules can be imported
sys.path.append(str(Path(__file__).parent))

from database import engine, Base, SessionLocal, get_db
from models.customer import PredictionHistoryModel
from schemas import CustomerInput, PredictionResponse, ExplainResponse, ShapFeature
from feature_engineering import payload_to_dataframe
from services.risk_service import classify_risk, get_recommendation
from routers import customers_router, analytics_router, predictions_router

app = FastAPI(
    title="Customer Churn Prediction & Retention API",
    description="REST API serving Customer Churn predictions, Customer Management, Analytics, and Retention Intelligence.",
    version="2.0.0"
)

# --- Configurable CORS Middleware ---
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
origins = [
    frontend_url,
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8501",  # Streamlit
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Modular Routers ---
app.include_router(customers_router)
app.include_router(analytics_router)
app.include_router(predictions_router)

# --- Global Model & Artifact State ---
model = None
threshold = 0.5
metadata = {}

# SHAP global state
_shap_explainer    = None   # shap.TreeExplainer fitted on the raw GBM step
_shap_feature_names: list  = []
_shap_expected_value: float = 0.0


def _load_shap_explainer(gbm_pipeline):
    """Initialise the SHAP TreeExplainer from the saved GBM imblearn pipeline."""
    global _shap_explainer, _shap_expected_value
    try:
        # The pipeline may use step name 'model' (train_model.py) or 'm' (_retrain.py)
        step_name = 'model' if 'model' in gbm_pipeline.named_steps else 'm'
        raw_model = gbm_pipeline.named_steps[step_name]
        _shap_explainer = shap.TreeExplainer(raw_model)
        ev = _shap_explainer.expected_value
        # expected_value is an array for binary classifiers in some SHAP versions
        if hasattr(ev, '__len__'):
            _shap_expected_value = float(ev[-1])
        else:
            _shap_expected_value = float(ev)
        print("[+] SHAP TreeExplainer initialised.", flush=True)
    except Exception as e:
        print(f"[!] SHAP explainer init failed: {e}", flush=True)


# --- Load Model Artifact on Startup ---
def load_model_artifact():
    global model, threshold, metadata, _shap_feature_names

    possible_paths = [
        Path('backend/model.pkl'),
        Path('model.pkl'),
        Path(__file__).parent.parent / 'model.pkl'
    ]

    model_file = None
    for p in possible_paths:
        if p.exists():
            model_file = p
            break

    if model_file is None:
        print("[!] Warning: model.pkl artifact file not found.", flush=True)
        return

    try:
        with open(model_file, 'rb') as f:
            artifact = pickle.load(f)

        if isinstance(artifact, dict) and 'model' in artifact:
            model = artifact['model']
            threshold = float(artifact.get('threshold', 0.5))
            metadata = artifact.get('metadata', {})
            if "model_version" not in metadata:
                metadata["model_version"] = "v1.0"
            # Load SHAP feature names from metadata if present
            _shap_feature_names = metadata.get('feature_names', [])
            # Boot SHAP explainer from the pipeline
            _load_shap_explainer(model)
            print(f"[+] Loaded model artifact from {model_file}", flush=True)
            print(f"    Dynamic Threshold: {threshold:.2f}", flush=True)
            print(f"    Metadata: {metadata}", flush=True)
        else:
            # Fallback for raw model objects
            model = artifact
            threshold = 0.5
            metadata = {"model_version": "v1.0"}
            print(f"[+] Loaded legacy model object from {model_file} (Default threshold: 0.5)", flush=True)

    except Exception as e:
        model = None
        threshold = 0.5
        metadata = {"model_version": "v1.0"}
        print(f"[-] Error loading model artifact: {e}", flush=True)


# Init database tables if engine is connected
if engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"⚠️ Table creation warning: {e}", flush=True)

# Call loader on initialization — run in a background thread so SHAP
# TreeExplainer initialisation does not block the uvicorn startup loop.
import threading
threading.Thread(target=load_model_artifact, daemon=True).start()


# --- SHAP Helper ---
def _compute_shap_for_input(df_single) -> tuple[list, float]:
    """
    Compute SHAP values for a single-row DataFrame.
    Returns (list of ShapFeature dicts sorted by |shap_value| desc, expected_value).
    """
    if _shap_explainer is None:
        return [], _shap_expected_value

    # Step name may be 'preprocessor' (train_model.py) or 'pre' (_retrain.py)
    pre_step = 'preprocessor' if 'preprocessor' in model.named_steps else 'pre'
    preprocessor = model.named_steps[pre_step]
    X_transformed = preprocessor.transform(df_single)

    # feature names: prefer metadata list, fallback to index labels
    feat_names = _shap_feature_names if _shap_feature_names else [
        f"feature_{i}" for i in range(X_transformed.shape[1])
    ]

    shap_vals = _shap_explainer.shap_values(X_transformed)[0]  # shape (n_features,)

    pairs = sorted(
        zip(feat_names, shap_vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return [{"feature": f, "shap_value": round(v, 6)} for f, v in pairs], _shap_expected_value


# --- Core Endpoints ---
@app.get("/health")
async def health():
    """Health check endpoint returning API status and model metadata."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "shap_enabled": _shap_explainer is not None,
        "threshold": threshold,
        "metadata": metadata
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerInput, db: Session = Depends(get_db)):
    """Predict customer churn probability, status, risk level, retention recommendation, and top SHAP features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    try:
        # Convert API input payload into DataFrame via shared feature engineering
        payload_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        df = payload_to_dataframe(payload_dict)

        # Predict probability using pipeline
        proba = float(model.predict_proba(df)[0, 1])

        # Apply dynamically loaded threshold
        prediction = "Yes" if proba >= threshold else "No"

        # Categorize business risk level & retention recommendation
        risk_lvl = classify_risk(proba)
        recommended_act = get_recommendation(payload_dict, risk_lvl)
        m_version = str(metadata.get("model_version", "v1.0"))
        c_id = payload_dict.get("customer_id")

        # Inline SHAP — top-5 features for this prediction
        shap_features, _ = _compute_shap_for_input(df)
        top_shap = shap_features[:5]

        # Record prediction in history table if DB session is active
        if db is not None:
            try:
                history_record = PredictionHistoryModel(
                    customer_id=c_id,
                    probability=round(proba, 4),
                    prediction=prediction,
                    risk_level=risk_lvl,
                    threshold=threshold,
                    model_version=m_version
                )
                db.add(history_record)
                db.commit()
            except Exception as db_err:
                print(f"⚠️ Warning: Failed to record prediction history: {db_err}", flush=True)
                db.rollback()

        return {
            "prediction": prediction,
            "probability": proba,
            "threshold": threshold,
            "risk_level": risk_lvl,
            "recommended_action": recommended_act,
            "model_version": m_version,
            "customer_id": c_id,
            "shap_top_features": top_shap
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {e}")


@app.post("/explain", response_model=ExplainResponse)
async def explain_prediction(data: CustomerInput):
    """
    Return full per-feature SHAP values for a customer input.
    Use this endpoint to power the Explainability tab in the frontend.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    if _shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer is not available. Re-train the model.")

    try:
        payload_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        df = payload_to_dataframe(payload_dict)

        proba = float(model.predict_proba(df)[0, 1])
        shap_features, exp_val = _compute_shap_for_input(df)

        return {
            "customer_id":       payload_dict.get("customer_id"),
            "probability":       proba,
            "expected_value":    exp_val,
            "shap_top_features": shap_features[:10],
            "shap_all_features": shap_features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation processing error: {e}")
