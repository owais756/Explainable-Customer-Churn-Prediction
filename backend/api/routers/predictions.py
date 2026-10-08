# backend/api/routers/predictions.py

import math
import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).parent.parent))

from database import get_db
from models.customer import PredictionHistoryModel
from schemas import PredictionHistoryListResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=PredictionHistoryListResponse)
def get_predictions_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    customer_id: Optional[str] = Query(None, description="Filter history by Customer ID"),
    db: Session = Depends(get_db)
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    query = db.query(PredictionHistoryModel)

    if customer_id and customer_id.strip():
        query = query.filter(PredictionHistoryModel.customer_id == customer_id.strip())

    total = query.count()
    total_pages = max(1, math.ceil(total / limit))

    db_predictions = query.order_by(
        PredictionHistoryModel.created_at.desc()
    ).offset((page - 1) * limit).limit(limit).all()

    predictions_list = []
    for p in db_predictions:
        p_dict = p.__dict__.copy()
        p_dict.pop('_sa_instance_state', None)
        if p_dict.get("created_at"):
            p_dict["created_at"] = str(p_dict["created_at"])
        predictions_list.append(p_dict)

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "predictions": predictions_list
    }
