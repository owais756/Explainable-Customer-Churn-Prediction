import os
from pathlib import Path

import pandas as pd

# Database configuration
DB_USER     = os.getenv('DB_USER', 'user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST     = os.getenv('DB_HOST', 'postgres')
DB_PORT     = os.getenv('DB_PORT', '5432')
DB_NAME     = os.getenv('DB_NAME', 'churn_db')

# Possible CSV fallback locations
_CSV_CANDIDATES = [
    Path(__file__).parent.parent / 'data' / 'Telco-Customer-Churn.csv',
    Path('data/Telco-Customer-Churn.csv'),
    Path('../data/Telco-Customer-Churn.csv'),
]

def load_data() -> pd.DataFrame:
    """
    Load customer data.
    Tries PostgreSQL first; falls back to local CSV if DB is unavailable.
    The SQLAlchemy engine is created lazily so a missing psycopg driver
    does not crash the app at startup.
    """
    # --- Try PostgreSQL ---
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
            connect_args={"connect_timeout": 3}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))   # quick reachability check
        df = pd.read_sql("SELECT * FROM customers", engine)
        print("[+] Frontend: data loaded from PostgreSQL.", flush=True)
        return df
    except Exception as db_err:
        print(f"[*] Frontend: PostgreSQL unavailable ({db_err}). Using local CSV.", flush=True)

    # --- CSV fallback ---
    for path in _CSV_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            # Normalise column names to match DB schema
            df = df.rename(columns={
                'customerID':       'customer_id',
                'SeniorCitizen':    'senior_citizen',
                'PhoneService':     'phone_service',
                'MultipleLines':    'multiple_lines',
                'InternetService':  'internet_service',
                'OnlineSecurity':   'online_security',
                'OnlineBackup':     'online_backup',
                'DeviceProtection': 'device_protection',
                'TechSupport':      'tech_support',
                'StreamingTV':      'streaming_tv',
                'StreamingMovies':  'streaming_movies',
                'PaperlessBilling': 'paperless_billing',
                'MonthlyCharges':   'monthly_charges',
                'TotalCharges':     'total_charges',
                'PaymentMethod':    'payment_method',
                'Churn':            'churn',
                'Partner':          'partner',
                'Dependents':       'dependents',
                'Contract':         'contract',
                'Gender':           'gender',
                'Tenure':           'tenure',
            })
            print(f"[+] Frontend: data loaded from CSV ({path}).", flush=True)
            return df

    raise FileNotFoundError(
        "Could not load data: PostgreSQL is unreachable and no local CSV was found.\n"
        "Expected CSV at: data/Telco-Customer-Churn.csv"
    )
