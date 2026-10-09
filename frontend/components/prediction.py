import streamlit as st
import requests
import os

# Try to read from the environment, defaulting to the Compose service name:
API_URL = os.getenv("API_URL", "http://localhost:8000")

def prediction_form():
    st.header("🔮 Customer Churn Prediction")
    st.markdown(
        "Fill in the customer’s profile below. "
        "Hover over any input for guidance on how to choose."
    )
    with st.form("prediction_form"):
        # --- Numeric Inputs ---
        tenure = st.number_input(
            "⏳ Tenure (months)",
            min_value=0, max_value=100, value=12,
            help="How many months this customer has been with the company."
        )
        monthly_charges = st.number_input(
            "💰 Monthly Charges ($)",
            min_value=0.0, value=70.0, format="%.2f",
            help="What the customer pays every month, on average."
        )

        # --- Core Categorical Inputs ---
        contract = st.selectbox(
            "📄 Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help=(
                "Month-to-month contracts often see higher churn.\n"
                "One- and two-year contracts generally retain customers longer."
            )
        )
        gender = st.selectbox(
            "👤 Gender",
            ["Female", "Male"],
            help="Select the customer’s reported gender."
        )
        senior_citizen = st.selectbox(
            "🧓 Senior Citizen",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether the customer is a senior citizen (1 = Yes)."
        )
        partner = st.selectbox(
            "💑 Has Partner?",
            ["Yes", "No"],
            help="Whether the customer has a partner."
        )
        dependents = st.selectbox(
            "👨‍👩‍👧 Has Dependents?",
            ["Yes", "No"],
            help="Whether the customer has dependents."
        )

        # --- Network & Billing Inputs ---
        internet_service = st.selectbox(
            "🌐 Internet Service",
            ["DSL", "Fiber optic", "No"],
            help=(
                "Type of internet service the customer uses. "
                "Customers with Fiber optic historically churn more."
            )
        )
        paperless_billing = st.selectbox(
            "🧾 Paperless Billing?",
            ["Yes", "No"],
            help="Whether the customer uses paperless billing."
        )
        payment_method = st.selectbox(
            "💳 Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            help=(
                "How the customer pays their bill. "
                "Electronic check users are often at higher churn risk."
            )
        )

        # --- Services Multiselect ---
        services_list = st.multiselect(
            "🛎️ Optional Services",
            ["Multiple Lines", "Online Security", "Online Backup",
             "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"],
            help=(
                "Choose all optional services the customer subscribes to. "
                "More services generally indicate lower churn risk."
            )
        )

        submitted = st.form_submit_button("Predict Churn")
        if submitted:
            payload = {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "contract": contract,
                "gender": gender,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
                "internet_service": internet_service,
                "paperless_billing": paperless_billing,
                "payment_method": payment_method,
                "services_list": services_list
            }
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    prob       = result['probability']
                    prediction = result['prediction']
                    risk_lvl   = result.get('risk_level', '')
                    rec_action = result.get('recommended_action', '')

                    # Colour-coded prediction banner
                    if prediction == "Yes":
                        st.error(
                            f"⚠️ **Prediction: CHURN**  \n"
                            f"Churn Probability: **{prob:.2%}**  |  Risk Level: **{risk_lvl}**"
                        )
                    else:
                        st.success(
                            f"✅ **Prediction: NO CHURN**  \n"
                            f"Churn Probability: **{prob:.2%}**  |  Risk Level: **{risk_lvl}**"
                        )

                    if rec_action:
                        st.info(f"💡 **Recommended Action:** {rec_action}")

                    # SHAP top features inline
                    shap_feats = result.get('shap_top_features', [])
                    if shap_feats:
                        st.markdown("#### 🔍 Top Factors Driving This Prediction")
                        import plotly.graph_objects as go
                        names  = [f["feature"]    for f in shap_feats]
                        values = [f["shap_value"] for f in shap_feats]
                        colors = ["#e53935" if v > 0 else "#1e88e5" for v in values]
                        fig = go.Figure(go.Bar(
                            x=values, y=names, orientation='h',
                            marker_color=colors,
                            text=[f"{v:+.4f}" for v in values],
                            textposition='outside'
                        ))
                        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                        fig.update_layout(
                            xaxis_title="SHAP Value",
                            yaxis=dict(autorange="reversed"),
                            height=260, template="plotly_white",
                            margin=dict(l=10, r=50, t=10, b=30)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("🔴 Red = pushes towards churn · 🔵 Blue = reduces churn risk")
                else:
                    st.error(f"Prediction failed ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
