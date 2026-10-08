# backend/api/main.py

import os
import sys
import pickle
from pathlib import Path
from typing import List, Optional
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Ensure local api modules can be imported
sys.path.append(str(Path(__file__).parent))

from database import engine, Base, SessionLocal, get_db
from models.customer import PredictionHistoryModel
from schemas import CustomerInput, PredictionResponse
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


# --- Load Model Artifact on Startup ---
def load_model_artifact():
    global model, threshold, metadata

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

# Call loader on initialization
load_model_artifact()


# --- Core Endpoints ---
@app.get("/health")
async def health():
    """Health check endpoint returning API status and model metadata."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": threshold,
        "metadata": metadata
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerInput, db: Session = Depends(get_db)):
    """Predict customer churn probability, status, risk level, and retention recommendation."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    try:
        # Convert API input payload into DataFrame via shared feature engineering
        payload_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        df = payload_to_dataframe(payload_dict)

        # Predict probability using pipeline
        proba = float(model.predict_proba(df)[0, 1])

        # Apply dynamically loaded threshold (0.33)
        prediction = "Yes" if proba >= threshold else "No"

        # Categorize business risk level & retention recommendation
        risk_lvl = classify_risk(proba)
        recommended_act = get_recommendation(payload_dict, risk_lvl)
        m_version = str(metadata.get("model_version", "v1.0"))
        c_id = payload_dict.get("customer_id")

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
            "customer_id": c_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {e}")
