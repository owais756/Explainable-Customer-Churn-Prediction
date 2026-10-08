# backend/api/schemas.py

from typing import List, Optional
from pydantic import BaseModel, conint, confloat, constr, ConfigDict


class CustomerInput(BaseModel):
    customer_id:       Optional[str] = None
    tenure:            conint(ge=0)
    monthly_charges:   confloat(ge=0.0)
    contract:          constr(min_length=1)
    gender:            constr(min_length=1)
    senior_citizen:    conint(ge=0, le=1)
    partner:           constr(min_length=1)
    dependents:        constr(min_length=1)

    internet_service:  constr(min_length=1)
    paperless_billing: constr(min_length=1)
    payment_method:    constr(min_length=1)

    # Optional total_charges field
    total_charges:     Optional[confloat(ge=0.0)] = None

    # List of selected service names
    services_list:     List[constr(min_length=1)] = []


class PredictionResponse(BaseModel):
    prediction:          str
    probability:         float
    threshold:           float
    risk_level:          str
    recommended_action:  str
    model_version:       str
    customer_id:         Optional[str] = None


class CustomerDetail(BaseModel):
    customer_id:         str
    gender:              Optional[str] = None
    senior_citizen:      Optional[int] = None
    partner:             Optional[str] = None
    dependents:          Optional[str] = None
    tenure:              Optional[int] = None
    phone_service:       Optional[str] = None
    multiple_lines:      Optional[str] = None
    internet_service:    Optional[str] = None
    online_security:     Optional[str] = None
    online_backup:       Optional[str] = None
    device_protection:   Optional[str] = None
    tech_support:        Optional[str] = None
    streaming_tv:        Optional[str] = None
    streaming_movies:    Optional[str] = None
    contract:            Optional[str] = None
    paperless_billing:   Optional[str] = None
    payment_method:      Optional[str] = None
    monthly_charges:     Optional[float] = None
    total_charges:       Optional[float] = None
    churn:               Optional[str] = None
    churn_probability:   Optional[float] = None
    risk_level:          Optional[str] = None
    recommended_action:  Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CustomerListResponse(BaseModel):
    total:       int
    page:        int
    limit:       int
    total_pages: int
    customers:   List[CustomerDetail]


class AnalyticsOverview(BaseModel):
    total_customers:     int
    churned_customers:   int
    churn_rate:          float
    high_risk_count:     int
    medium_risk_count:   int
    low_risk_count:      int
    revenue_at_risk:     float
    model_version:       str
    threshold:           float


class PredictionHistoryItem(BaseModel):
    id:            int
    customer_id:   Optional[str] = None
    probability:   float
    prediction:    str
    risk_level:    str
    threshold:     float
    model_version: str
    created_at:    Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionHistoryListResponse(BaseModel):
    total:       int
    page:        int
    limit:       int
    total_pages: int
    predictions: List[PredictionHistoryItem]
