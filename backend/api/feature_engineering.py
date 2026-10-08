# backend/api/feature_engineering.py

import numpy as np
import pandas as pd

# List of service columns used to calculate services_count
SERVICE_COLS = [
    'multiple_lines',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies'
]

# Mapping from API service names to DataFrame column names
SERVICE_MAP = {
    "Multiple Lines":     "multiple_lines",
    "Online Security":    "online_security",
    "Online Backup":      "online_backup",
    "Device Protection":  "device_protection",
    "Tech Support":       "tech_support",
    "Streaming TV":       "streaming_tv",
    "Streaming Movies":   "streaming_movies",
}

# Feature columns order expected by the preprocessing pipeline
FEATURE_COLUMNS = [
    'tenure', 'monthly_charges', 'total_charges', 'avg_charge_per_month',
    'services_count', 'contract', 'gender', 'senior_citizen', 'partner',
    'dependents', 'internet_service', 'paperless_billing', 'payment_method',
    'multiple_lines', 'online_security', 'online_backup',
    'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
    'tenure_bucket'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies unified feature engineering on raw customer DataFrame.
    Derived features created:
      1. avg_charge_per_month = total_charges / (tenure + 1e-3)
      2. services_count = sum of active services ('yes')
      3. tenure_bucket = categorical bucket (0-12, 12-24, 24-48, 48+)
    """
    df = df.copy()

    # 1. Total charges numeric & avg charge per month
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['avg_charge_per_month'] = df['total_charges'] / (df['tenure'] + 1e-3)

    # 2. Services count
    # Ensure service columns exist
    for col in SERVICE_COLS:
        if col not in df.columns:
            df[col] = "No"

    df['services_count'] = df[SERVICE_COLS].apply(
        lambda row: sum(str(x).strip().lower() == 'yes' for x in row),
        axis=1
    )

    # 3. Tenure bucket
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, np.inf],
        labels=['0-12', '12-24', '24-48', '48+']
    )

    return df


def payload_to_dataframe(data_dict: dict) -> pd.DataFrame:
    """
    Converts a single API payload dictionary (from CustomerInput schema)
    into a feature DataFrame ready for model prediction.
    """
    d = data_dict.copy()

    # Derive total_charges if not provided directly
    if 'total_charges' not in d or d['total_charges'] is None:
        d['total_charges'] = float(d['tenure'] * d['monthly_charges'])

    # Reconstruct service flags from services_list
    services_list = d.pop('services_list', [])
    for col in SERVICE_COLS:
        d[col] = "No"

    for svc in services_list:
        col_name = SERVICE_MAP.get(svc)
        if col_name:
            d[col_name] = "Yes"

    df_single = pd.DataFrame([d])
    df_engineered = engineer_features(df_single)

    # Return DataFrame with exact column ordering
    return df_engineered[FEATURE_COLUMNS]
