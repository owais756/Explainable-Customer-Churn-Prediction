# frontend/components/explainability.py
"""
SHAP-based Explainability Tab
-------------------------------
Renders two sub-sections:
  1. Global Feature Importance  — mean |SHAP| bar chart loaded from backend/reports/
  2. Individual Explanation     — per-customer waterfall-style bar chart via /explain API
"""

import os
import io
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Possible locations for the SHAP bar image saved during training
_SHAP_BAR_CANDIDATES = [
    Path("backend/reports/shap_summary_bar.png"),
    Path("reports/shap_summary_bar.png"),
]
_SHAP_BEE_CANDIDATES = [
    Path("backend/reports/shap_summary_beeswarm.png"),
    Path("reports/shap_summary_beeswarm.png"),
]


def _find_image(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def _waterfall_chart(shap_features: list, expected_value: float, probability: float, title: str):
    """
    Build a Plotly horizontal waterfall chart showing how each SHAP value
    pushes the model output away from the expected (base) value.
    """
    # Keep top-12 by absolute SHAP, already sorted desc by API
    features = shap_features[:12]
    names  = [f["feature"]    for f in features]
    values = [f["shap_value"] for f in features]

    # Running total for hover text
    running = expected_value
    totals  = []
    for v in values:
        running += v
        totals.append(round(running, 4))

    colors = ["#e53935" if v > 0 else "#1e88e5" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP: %{x:.4f}<br>"
            "Cumulative: %{customdata:.4f}<extra></extra>"
        ),
        customdata=totals
    ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(autorange="reversed"),
        height=max(350, len(features) * 30 + 100),
        template="plotly_white",
        margin=dict(l=20, r=60, t=60, b=40),
        annotations=[
            dict(
                x=0.01, y=1.04, xref="paper", yref="paper",
                text=f"Base value: {expected_value:.4f}  →  Final prob: {probability:.4f}",
                showarrow=False, font=dict(size=12, color="#555")
            )
        ]
    )
    return fig


def explainability_tab(df: pd.DataFrame):
    st.header("🔍 SHAP Explainability")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** decomposes each prediction into the contribution of every feature.  
    - 🔴 **Red bars** push the prediction *towards churn* (positive SHAP)  
    - 🔵 **Blue bars** push the prediction *away from churn* (negative SHAP)  
    - The **base value** is the average model output across all training samples.
    """)

    tab_global, tab_individual = st.tabs(["🌐 Global Feature Importance", "👤 Individual Explanation"])

    # ── Global tab ─────────────────────────────────────────────────────────────
    with tab_global:
        st.subheader("Global SHAP Feature Importance")
        st.markdown(
            "The charts below are generated during model training and show which features "
            "have the largest average impact on churn predictions across the test set."
        )

        bar_path = _find_image(_SHAP_BAR_CANDIDATES)
        bee_path = _find_image(_SHAP_BEE_CANDIDATES)

        if bar_path:
            st.image(str(bar_path), caption="Mean |SHAP| — Top 20 Features", use_container_width=True)
        else:
            st.info("Global SHAP bar chart not found. Run `train_model.py` to generate it.")

        if bee_path:
            with st.expander("🐝 SHAP Beeswarm Plot (feature direction & spread)"):
                st.image(str(bee_path), caption="SHAP Beeswarm — Impact direction per sample", use_container_width=True)
                st.markdown("""
                Each dot is one customer from the test set. The colour indicates the raw feature value
                (red = high, blue = low). The x-axis shows the SHAP impact on the churn score.
                """)
        else:
            st.info("Beeswarm plot not found. Re-train the model to generate it.")

        # Interactive global importance from metadata via /health
        st.markdown("---")
        st.subheader("Live Feature Importance from Loaded Model")
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            if resp.status_code == 200:
                meta = resp.json().get("metadata", {})
                feat_names  = meta.get("feature_names", [])
                mean_shap   = meta.get("mean_abs_shap", [])
                if feat_names and mean_shap:
                    pairs = sorted(
                        zip(feat_names, mean_shap),
                        key=lambda x: x[1], reverse=True
                    )[:20]
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in pairs],
                        y=[n for n, _ in pairs],
                        orientation='h',
                        marker_color='#3b82d4',
                        text=[f"{v:.4f}" for _, v in pairs],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Mean |SHAP| — Top 20 (from live model metadata)",
                        xaxis_title="Mean |SHAP Value|",
                        yaxis=dict(autorange="reversed"),
                        height=520,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available in model metadata. Re-train to enable.")
            else:
                st.warning(f"Backend health check failed ({resp.status_code}).")
        except Exception as e:
            st.warning(f"Could not reach the API to fetch live metadata: {e}")

    # ── Individual tab ──────────────────────────────────────────────────────────
    with tab_individual:
        st.subheader("Individual Customer Explanation")
        st.markdown(
            "Fill in a customer profile and click **Explain** to see exactly which features "
            "drive that specific prediction."
        )

        with st.form("shap_explain_form"):
            col1, col2 = st.columns(2)
            with col1:
                tenure          = st.number_input("⏳ Tenure (months)",        min_value=0,  max_value=100, value=12)
                monthly_charges = st.number_input("💰 Monthly Charges ($)",    min_value=0.0, value=70.0, format="%.2f")
                contract        = st.selectbox("📄 Contract",                  ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("🌐 Internet Service",          ["DSL", "Fiber optic", "No"])
                payment_method  = st.selectbox("💳 Payment Method",            ["Electronic check", "Mailed check",
                                                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
            with col2:
                gender          = st.selectbox("👤 Gender",                    ["Female", "Male"])
                senior_citizen  = st.selectbox("🧓 Senior Citizen",            [0, 1],
                                               format_func=lambda x: "No" if x == 0 else "Yes")
                partner         = st.selectbox("💑 Has Partner?",              ["Yes", "No"])
                dependents      = st.selectbox("👨‍👩‍👧 Has Dependents?",           ["Yes", "No"])
                paperless_billing = st.selectbox("🧾 Paperless Billing?",      ["Yes", "No"])
                services_list   = st.multiselect("🛎️ Optional Services",
                                                  ["Multiple Lines", "Online Security", "Online Backup",
                                                   "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"])

            explain_btn = st.form_submit_button("🔍 Explain Prediction")

        if explain_btn:
            payload = {
                "tenure": tenure, "monthly_charges": monthly_charges,
                "contract": contract, "gender": gender,
                "senior_citizen": senior_citizen, "partner": partner,
                "dependents": dependents, "internet_service": internet_service,
                "paperless_billing": paperless_billing,
                "payment_method": payment_method, "services_list": services_list
            }
            with st.spinner("Computing SHAP values..."):
                try:
                    resp = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
                    if resp.status_code == 200:
                        result = resp.json()
                        prob     = result["probability"]
                        exp_val  = result["expected_value"]
                        all_feats = result["shap_all_features"]

                        label_color = "🔴" if prob >= 0.5 else "🟢"
                        st.markdown(
                            f"### {label_color} Churn Probability: **{prob:.2%}**  "
                            f"&nbsp;&nbsp; *(base value: {exp_val:.4f})*"
                        )

                        fig = _waterfall_chart(
                            all_feats, exp_val, prob,
                            title="SHAP Waterfall — Feature Contributions to Churn Prediction"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary table
                        top10 = result["shap_top_features"]
                        table_df = pd.DataFrame(top10).rename(
                            columns={"feature": "Feature", "shap_value": "SHAP Value"}
                        )
                        table_df["Direction"] = table_df["SHAP Value"].apply(
                            lambda v: "↑ Increases churn risk" if v > 0 else "↓ Reduces churn risk"
                        )
                        st.dataframe(table_df, use_container_width=True, hide_index=True)

                    elif resp.status_code == 503:
                        st.error(
                            "SHAP explainer is not available. "
                            "Re-train the model with `train_model.py` to enable per-customer SHAP."
                        )
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Could not reach the prediction API: {e}")
