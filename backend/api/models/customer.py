# backend/api/models/customer.py

from sqlalchemy import Column, String, Integer, Numeric, DateTime, func
import sys
from pathlib import Path

# Ensure database module can be imported
sys.path.append(str(Path(__file__).parent.parent))
from database import Base


class CustomerModel(Base):
    __tablename__ = "customers"

    customer_id = Column(String(50), primary_key=True, index=True)
    gender = Column(String(10))
    senior_citizen = Column(Integer)
    partner = Column(String(10))
    dependents = Column(String(10))
    tenure = Column(Integer)
    phone_service = Column(String(10))
    multiple_lines = Column(String(20))
    internet_service = Column(String(20))
    online_security = Column(String(20))
    online_backup = Column(String(20))
    device_protection = Column(String(20))
    tech_support = Column(String(20))
    streaming_tv = Column(String(20))
    streaming_movies = Column(String(20))
    contract = Column(String(20))
    paperless_billing = Column(String(10))
    payment_method = Column(String(50))
    monthly_charges = Column(Numeric(10, 2))
    total_charges = Column(Numeric(10, 2))
    churn = Column(String(10))


class PredictionHistoryModel(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(String(50), nullable=True, index=True)
    probability = Column(Numeric(5, 4))
    prediction = Column(String(10))
    risk_level = Column(String(10))
    threshold = Column(Numeric(5, 4))
    model_version = Column(String(20))
    created_at = Column(DateTime, server_default=func.now())
