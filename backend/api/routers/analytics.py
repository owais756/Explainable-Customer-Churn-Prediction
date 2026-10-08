# backend/api/routers/analytics.py

import sys
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).parent.parent))

from database import get_db
from models.customer import CustomerModel
from schemas import AnalyticsOverview
from services.risk_service import classify_risk
from feature_engineering import engineer_features, FEATURE_COLUMNS

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverview)
def get_analytics_overview(db: Session = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    from main import model, threshold, metadata

    customers = db.query(CustomerModel).all()
    total_customers = len(customers)

    if total_customers == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "churn_rate": 0.0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "revenue_at_risk": 0.0,
            "model_version": str(metadata.get("model_version", "v1.0")),
            "threshold": threshold
        }

    # Count churned from DB column
    churned_customers = sum(
        1 for c in customers if str(c.churn or "").strip().lower() == "yes"
    )
    churn_rate = round(churned_customers / total_customers, 4)

    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    revenue_at_risk = 0.0

    if model is not None:
        c_dicts = [c.__dict__.copy() for c in customers]
        for cd in c_dicts:
            cd.pop('_sa_instance_state', None)

        try:
            df_all = pd.DataFrame(c_dicts)
            df_eng = engineer_features(df_all)[FEATURE_COLUMNS]
            probas = model.predict_proba(df_eng)[:, 1]

            for i, cd in enumerate(c_dicts):
                p = float(probas[i])
                risk = classify_risk(p)
                m_charge = float(cd.get("monthly_charges") or 0.0)

                if risk == "HIGH":
                    high_risk_count += 1
                    revenue_at_risk += m_charge
                elif risk == "MEDIUM":
                    medium_risk_count += 1
                else:
                    low_risk_count += 1

        except Exception:
            # Fallback if evaluation fails
            pass

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "churn_rate": churn_rate,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "model_version": str(metadata.get("model_version", "v1.0")),
        "threshold": threshold
    }
