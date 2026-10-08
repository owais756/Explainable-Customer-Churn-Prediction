# backend/api/routers/customers.py

import math
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

sys.path.append(str(Path(__file__).parent.parent))

from database import get_db
from models.customer import CustomerModel, PredictionHistoryModel
from schemas import CustomerDetail, CustomerListResponse, PredictionHistoryItem
from services.risk_service import classify_risk, get_recommendation
from feature_engineering import engineer_features, FEATURE_COLUMNS

router = APIRouter(prefix="/customers", tags=["customers"])


def _eval_customer_risk(c_dict: dict, model) -> dict:
    """Helper to append churn_probability, risk_level, and recommended_action to a customer dict."""
    if model is None:
        c_dict["churn_probability"] = None
        c_dict["risk_level"] = "UNKNOWN"
        c_dict["recommended_action"] = "N/A"
        return c_dict

    try:
        df_single = pd.DataFrame([c_dict])
        df_eng = engineer_features(df_single)[FEATURE_COLUMNS]
        proba = float(model.predict_proba(df_eng)[0, 1])
        risk = classify_risk(proba)
        rec = get_recommendation(c_dict, risk)

        c_dict["churn_probability"] = proba
        c_dict["risk_level"] = risk
        c_dict["recommended_action"] = rec
    except Exception:
        c_dict["churn_probability"] = None
        c_dict["risk_level"] = "UNKNOWN"
        c_dict["recommended_action"] = "N/A"

    return c_dict


def _eval_customers_batch(c_dicts: List[dict], model) -> List[dict]:
    """Batch evaluate risk and recommendations for a list of customer dicts."""
    if not c_dicts or model is None:
        for cd in c_dicts:
            cd["churn_probability"] = None
            cd["risk_level"] = "UNKNOWN"
            cd["recommended_action"] = "N/A"
        return c_dicts

    try:
        df_all = pd.DataFrame(c_dicts)
        df_eng = engineer_features(df_all)[FEATURE_COLUMNS]
        probas = model.predict_proba(df_eng)[:, 1]

        for i, cd in enumerate(c_dicts):
            p = float(probas[i])
            r = classify_risk(p)
            rec = get_recommendation(cd, r)
            cd["churn_probability"] = p
            cd["risk_level"] = r
            cd["recommended_action"] = rec
    except Exception:
        for cd in c_dicts:
            cd["churn_probability"] = None
            cd["risk_level"] = "UNKNOWN"
            cd["recommended_action"] = "N/A"

    return c_dicts


@router.get("", response_model=CustomerListResponse)
def get_customers(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by Customer ID"),
    risk: Optional[str] = Query(None, description="Filter by risk level: low, medium, high"),
    db: Session = Depends(get_db)
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    from main import model

    query = db.query(CustomerModel)

    if search:
        query = query.filter(CustomerModel.customer_id.ilike(f"%{search.strip()}%"))

    # If risk filter is provided, we must evaluate risk on matching set
    if risk and risk.strip():
        target_risk = risk.strip().upper()

        # Fetch matching items to filter by risk in Python
        raw_customers = query.all()
        c_dicts = [c.__dict__.copy() for c in raw_customers]
        for cd in c_dicts:
            cd.pop('_sa_instance_state', None)

        evaluated = _eval_customers_batch(c_dicts, model)
        filtered = [c for c in evaluated if c.get("risk_level") == target_risk]

        total = len(filtered)
        total_pages = max(1, math.ceil(total / limit))
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_customers = filtered[start_idx:end_idx]

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "customers": page_customers
        }

    # Standard database pagination
    total = query.count()
    total_pages = max(1, math.ceil(total / limit))

    db_customers = query.offset((page - 1) * limit).limit(limit).all()
    c_dicts = [c.__dict__.copy() for c in db_customers]
    for cd in c_dicts:
        cd.pop('_sa_instance_state', None)

    page_customers = _eval_customers_batch(c_dicts, model)

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "customers": page_customers
    }


@router.get("/{customer_id}", response_model=CustomerDetail)
def get_customer_by_id(customer_id: str, db: Session = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    from main import model

    customer = db.query(CustomerModel).filter(CustomerModel.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    c_dict = customer.__dict__.copy()
    c_dict.pop('_sa_instance_state', None)

    return _eval_customer_risk(c_dict, model)


@router.get("/{customer_id}/predictions", response_model=List[PredictionHistoryItem])
def get_customer_predictions(customer_id: str, db: Session = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    # Check if customer exists in customers table or prediction history
    customer_exists = db.query(CustomerModel).filter(CustomerModel.customer_id == customer_id).first()
    history = db.query(PredictionHistoryModel).filter(
        PredictionHistoryModel.customer_id == customer_id
    ).order_by(PredictionHistoryModel.created_at.desc()).all()

    if not customer_exists and not history:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    result = []
    for h in history:
        item = h.__dict__.copy()
        item.pop('_sa_instance_state', None)
        if item.get("created_at"):
            item["created_at"] = str(item["created_at"])
        result.append(item)

    return result
