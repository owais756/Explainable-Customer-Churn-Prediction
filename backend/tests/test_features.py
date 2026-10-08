# backend/tests/test_features.py

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add backend and backend/api to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))
sys.path.append(str(backend_dir / 'api'))

from api.feature_engineering import engineer_features, payload_to_dataframe, FEATURE_COLUMNS

def test_engineer_features():
    raw_data = pd.DataFrame([{
        'tenure': 10,
        'monthly_charges': 50.0,
        'total_charges': 500.0,
        'multiple_lines': 'Yes',
        'online_security': 'No',
        'online_backup': 'Yes',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No'
    }])

    df_out = engineer_features(raw_data)

    assert 'avg_charge_per_month' in df_out.columns
    assert 'services_count' in df_out.columns
    assert 'tenure_bucket' in df_out.columns

    # Check math: 500 / (10 + 0.001) = 49.995
    assert abs(df_out['avg_charge_per_month'].iloc[0] - 50.0) < 0.1
    # Check services count: 'Yes' on multiple_lines and online_backup -> 2
    assert df_out['services_count'].iloc[0] == 2
    # Check tenure bucket: 10 <= 12 -> '0-12'
    assert str(df_out['tenure_bucket'].iloc[0]) == '0-12'

def test_payload_to_dataframe():
    payload = {
        "tenure": 24,
        "monthly_charges": 80.0,
        "contract": "One year",
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "internet_service": "DSL",
        "paperless_billing": "No",
        "payment_method": "Mailed check",
        "services_list": ["Multiple Lines", "Tech Support"]
    }

    df_out = payload_to_dataframe(payload)

    assert list(df_out.columns) == FEATURE_COLUMNS
    assert df_out.shape == (1, 21)
    assert df_out['total_charges'].iloc[0] == 1920.0
    assert df_out['multiple_lines'].iloc[0] == 'Yes'
    assert df_out['tech_support'].iloc[0] == 'Yes'
    assert df_out['online_security'].iloc[0] == 'No'
    assert df_out['services_count'].iloc[0] == 2
    assert str(df_out['tenure_bucket'].iloc[0]) == '12-24'
