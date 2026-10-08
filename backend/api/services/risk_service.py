# backend/api/services/risk_service.py

def classify_risk(probability: float) -> str:
    """
    Deterministic risk classification:
    probability < 0.33        -> LOW
    0.33 <= probability < 0.66 -> MEDIUM
    probability >= 0.66       -> HIGH

    Note: Prediction threshold remains 0.33. The 0.66 value is solely
    for UI and business risk categorization.
    """
    if probability < 0.33:
        return "LOW"
    elif probability < 0.66:
        return "MEDIUM"
    else:
        return "HIGH"


def get_recommendation(customer_data: dict, risk_level: str) -> str:
    """
    Deterministic rule-based recommendation system.
    Rules:
      - HIGH risk + Month-to-month contract -> "Offer annual contract upgrade"
      - HIGH risk + no Tech Support        -> "Offer technical support"
      - HIGH risk + high monthly charges   -> "Consider targeted retention discount"
      - HIGH risk (other)                 -> "Assign dedicated retention specialist"
      - MEDIUM risk                       -> "Launch customer engagement campaign"
      - LOW risk                          -> "No immediate retention action required"
    """
    if risk_level == "HIGH":
        contract = str(customer_data.get("contract", "")).strip()
        tech_sup = str(customer_data.get("tech_support", "")).strip().lower()
        monthly_charges = float(customer_data.get("monthly_charges", 0.0) or 0.0)

        if contract == "Month-to-month":
            return "Offer annual contract upgrade"
        elif tech_sup in ["no", "no internet service"]:
            return "Offer technical support"
        elif monthly_charges >= 70.0:
            return "Consider targeted retention discount"
        else:
            return "Assign dedicated retention specialist"

    elif risk_level == "MEDIUM":
        return "Launch customer engagement campaign"

    else:
        return "No immediate retention action required"
