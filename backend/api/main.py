# backend/api/main.py

import sys
from pathlib import Path
from typing import List, Optional
import pickle
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat, constr

# Ensure local api modules can be imported
sys.path.append(str(Path(__file__).parent))
from feature_engineering import payload_to_dataframe

app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API serving Customer Churn predictions using an optimized Gradient Boosting Pipeline.",
    version="2.0.0"
)

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
            print(f"[+] Loaded model artifact from {model_file}", flush=True)
            print(f"    Dynamic Threshold: {threshold:.2f}", flush=True)
            print(f"    Metadata: {metadata}", flush=True)
        else:
            # Fallback for raw model objects
            model = artifact
            threshold = 0.5
            metadata = {}
            print(f"[+] Loaded legacy model object from {model_file} (Default threshold: 0.5)", flush=True)
            
    except Exception as e:
        model = None
        threshold = 0.5
        metadata = {}
        print(f"[-] Error loading model artifact: {e}", flush=True)

# Call loader on initialization
load_model_artifact()

# --- Define Input Schema ---
class CustomerInput(BaseModel):
    tenure:            conint(ge=0)
    monthly_charges:   confloat(ge=0.0)
    contract:          constr(strict=True)
    gender:            constr(strict=True)
    senior_citizen:    conint(ge=0, le=1)
    partner:           constr(strict=True)
    dependents:        constr(strict=True)

    internet_service:  constr(strict=True)
    paperless_billing: constr(strict=True)
    payment_method:    constr(strict=True)

    # Optional total_charges field
    total_charges:     Optional[confloat(ge=0.0)] = None

    # List of selected service names
    services_list:     List[constr(strict=True)] = []

# --- Endpoints ---
@app.get("/health")
async def health():
    """Health check endpoint returning API status and model metadata."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": threshold,
        "metadata": metadata
    }

@app.post("/predict")
async def predict_churn(data: CustomerInput):
    """Predict customer churn probability and status based on customer profile."""
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

        return {
            "prediction": prediction,
            "probability": proba,
            "threshold": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {e}")
