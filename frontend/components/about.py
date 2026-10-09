import streamlit as st

def about_tab():
    st.title("ℹ️ Explainable Customer Churn Prediction")
    st.caption("Machine Learning · SMOTE · SHAP · FastAPI · Streamlit")

    st.divider()

    # --- Overview ---
    st.subheader("📌 Project Overview")
    st.markdown(
        "This project builds an end-to-end explainable machine learning system to identify "
        "telecommunications customers at high risk of churn. It addresses class imbalance using "
        "**SMOTE**, evaluates multiple models using **Precision, Recall, F1-Score and ROC-AUC**, "
        "and uses **SHAP (SHapley Additive exPlanations)** to make every prediction interpretable — "
        "both globally across all customers and individually per prediction."
    )

    # --- Model Performance ---
    st.subheader("📊 Model Performance")
    import pandas as pd
    perf_df = pd.DataFrame([
        {"Model": "Logistic Regression (baseline)", "Imbalance Fix": "None",  "Threshold": "0.50",      "Precision": "—",     "Recall": "—",     "F1": "0.572", "ROC-AUC": "0.834"},
        {"Model": "Random Forest",                  "Imbalance Fix": "SMOTE", "Threshold": "Optimised", "Precision": "—",     "Recall": "—",     "F1": "0.625", "ROC-AUC": "0.831"},
        {"Model": "✅ Gradient Boosting (selected)", "Imbalance Fix": "SMOTE", "Threshold": "0.59",      "Precision": "0.580", "Recall": "0.682", "F1": "0.627", "ROC-AUC": "0.834"},
    ])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption("Dataset: Telco Customer Churn · 7,032 records · 2.8:1 class imbalance corrected by SMOTE")

    # --- SHAP Explainability ---
    st.subheader("🔬 SHAP Explainability")
    st.markdown(
        "Every prediction is explained using **SHAP TreeExplainer**:\n\n"
        "- **Global** — Mean |SHAP| bar chart and beeswarm plot show which features matter most across all customers.\n"
        "- **Individual** — The *Explainability* tab and the `/explain` API endpoint generate a waterfall chart "
        "for any customer, showing exactly which features push the prediction toward or away from churn."
    )
    st.markdown("**Top 5 churn drivers (globally):**")
    st.markdown(
        "1. `contract_Month-to-month` — strongest single predictor\n"
        "2. `tenure` — longer tenure significantly reduces churn\n"
        "3. `internet_service_Fiber optic` — Fiber optic users churn more\n"
        "4. `payment_method_Electronic check` — high-risk payment pattern\n"
        "5. `monthly_charges` — higher charges raise churn probability"
    )

    col_red, col_blue = st.columns(2)
    with col_red:
        st.error("🔴 Positive SHAP → pushes prediction **towards churn**")
    with col_blue:
        st.info("🔵 Negative SHAP → pushes prediction **away from churn**")

    # --- Dashboard Tabs ---
    st.subheader("🖥️ Dashboard Tabs")
    st.markdown(
        "| Tab | Description |\n"
        "|---|---|\n"
        "| 🏠 **Overview** | Distribution charts, demographic analysis, payment patterns, KMeans customer segmentation |\n"
        "| 📊 **Insights** | Churn rates by segment, correlation heatmap, actionable business findings |\n"
        "| 🔮 **Prediction** | Enter a customer profile → real-time churn prediction with risk level, recommended action, and inline SHAP chart |\n"
        "| 🔍 **Explainability** | Global SHAP bar + beeswarm charts and full per-customer waterfall explanation |\n"
        "| ℹ️ **About** | This page |"
    )

    # --- Tech Stack ---
    st.subheader("💻 Technology Stack")
    st.markdown(
        "| Layer | Technology |\n"
        "|---|---|\n"
        "| **ML & Explainability** | scikit-learn · imbalanced-learn (SMOTE) · SHAP |\n"
        "| **Backend API** | FastAPI · Uvicorn · Pydantic |\n"
        "| **Frontend** | Streamlit · Plotly |\n"
        "| **Database** | PostgreSQL · SQLAlchemy (local CSV fallback) |\n"
        "| **DevOps** | Docker · Docker Compose |"
    )

    # --- API Endpoints ---
    st.subheader("⚡ API Endpoints")
    st.markdown(
        "| Endpoint | Description |\n"
        "|---|---|\n"
        "| `POST /predict` | Prediction + probability + risk level + recommended action + top-5 SHAP features |\n"
        "| `POST /explain` | Full per-feature SHAP waterfall for any customer input |\n"
        "| `GET /health` | Model version, threshold, SHAP status, training metadata |\n"
        "| `GET /docs` | Interactive Swagger UI at http://localhost:8000/docs |"
    )

    st.divider()

    # --- Contact ---
    st.subheader("📧 Contact")
    st.markdown("**Owais Mohammed**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("📧 Email", "mailto:owaismohammed203@gmail.com", use_container_width=True)
    with col2:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/owais-mohammed-1ab936319", use_container_width=True)
    with col3:
        st.link_button("🐙 GitHub", "https://github.com/owais756", use_container_width=True)

    st.caption("⭐ If you find this project useful, please give it a star on GitHub!")
