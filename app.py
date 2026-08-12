"""
Phase 11: Dashboard (Enhanced)
==================================
A richer, portfolio-ready Streamlit dashboard with 5 tabs:
    1. Overview        -> dataset stats & distributions
    2. Credit Risk      -> scoring form + gauge chart
    3. Fraud Monitoring -> live table + fraud analytics charts
    4. Model Performance-> comparison charts, ROC curve, confusion matrix
    5. Feature Insights -> feature importance charts for both models

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models_saved"
OUT_DIR = BASE / "outputs"
DATA_DIR = BASE / "data"

st.set_page_config(page_title="Credit Risk & Fraud Detection", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    div[data-testid="stMetricValue"] {font-size: 28px;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Credit Risk Assessment & Fraud Detection System")
st.caption("End-to-end ML dashboard — scoring, monitoring, and model analytics")

# ---------------------------------------------------------------------------
# Load everything once
# ---------------------------------------------------------------------------
@st.cache_resource
def load_credit_artifacts():
    model = joblib.load(MODELS_DIR / "credit_risk_best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "credit_scaler.pkl")
    cols = joblib.load(MODELS_DIR / "credit_feature_columns.pkl")
    return model, scaler, cols


@st.cache_resource
def load_fraud_artifacts():
    model = joblib.load(MODELS_DIR / "fraud_best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "fraud_scaler.pkl")
    cols = joblib.load(MODELS_DIR / "fraud_feature_columns.pkl")
    return model, scaler, cols


@st.cache_data
def load_raw_data():
    credit_df = pd.read_csv(DATA_DIR / "credit_data.csv")
    txn_df = pd.read_csv(DATA_DIR / "transactions.csv")
    return credit_df, txn_df


@st.cache_data
def load_comparison_tables():
    credit_cmp = pd.read_csv(OUT_DIR / "credit_model_comparison.csv") if (OUT_DIR / "credit_model_comparison.csv").exists() else None
    fraud_cmp = pd.read_csv(OUT_DIR / "fraud_model_comparison.csv") if (OUT_DIR / "fraud_model_comparison.csv").exists() else None
    return credit_cmp, fraud_cmp


try:
    credit_model, credit_scaler, credit_cols = load_credit_artifacts()
    fraud_model, fraud_scaler, fraud_cols = load_fraud_artifacts()
    models_ready = True
except FileNotFoundError:
    models_ready = False
    st.error("⚠️ Models not found. Run `python main.py` first to train and save models, then reload this page.")

if models_ready:
    credit_df, txn_df = load_raw_data()
    credit_cmp, fraud_cmp = load_comparison_tables()

    tab_overview, tab_credit, tab_fraud, tab_perf, tab_features = st.tabs(
        ["📊 Overview", "💳 Credit Risk Scoring", "🚨 Fraud Monitoring",
         "📈 Model Performance", "🔍 Feature Insights"]
    )

    # =======================================================================
    # TAB 1: OVERVIEW
    # =======================================================================
    with tab_overview:
        st.header("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Applicants", f"{len(credit_df):,}")
        c2.metric("Default Rate", f"{credit_df['default'].mean():.1%}")
        c3.metric("Total Transactions", f"{len(txn_df):,}")
        c4.metric("Fraud Rate", f"{txn_df['is_fraud'].mean():.2%}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                credit_df, names=credit_df["default"].map({0: "Paid Back", 1: "Defaulted"}),
                title="Loan Outcome Distribution", hole=0.45,
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.histogram(
                credit_df, x="annual_income", color=credit_df["default"].map({0: "Paid Back", 1: "Defaulted"}),
                nbins=40, title="Income Distribution by Loan Outcome",
                color_discrete_sequence=["#2ecc71", "#e74c3c"], barmode="overlay", opacity=0.7
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.pie(
                txn_df, names=txn_df["is_fraud"].map({0: "Legitimate", 1: "Fraud"}),
                title="Transaction Outcome Distribution", hole=0.45,
                color_discrete_sequence=["#3498db", "#e74c3c"]
            )
            st.plotly_chart(fig3, use_container_width=True)

            fraud_by_cat = txn_df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False).reset_index()
            fig4 = px.bar(
                fraud_by_cat, x="merchant_category", y="is_fraud",
                title="Fraud Rate by Merchant Category",
                labels={"is_fraud": "Fraud Rate", "merchant_category": "Category"},
                color="is_fraud", color_continuous_scale="Reds"
            )
            st.plotly_chart(fig4, use_container_width=True)

    # =======================================================================
    # TAB 2: CREDIT RISK SCORING
    # =======================================================================
    with tab_credit:
        st.header("Credit Risk Assessment")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 90, 35)
            income = st.number_input("Annual Income", 0.0, 1_000_000.0, 50000.0)
            employment_years = st.number_input("Employment Years", 0.0, 50.0, 5.0)
            num_accounts = st.number_input("Number of Accounts", 0, 30, 4)
        with col2:
            credit_age = st.number_input("Credit Age (years)", 0.0, 40.0, 7.0)
            existing_debt = st.number_input("Existing Debt", 0.0, 500000.0, 8000.0)
            loan_amount = st.number_input("Requested Loan Amount", 0.0, 500000.0, 10000.0)
        with col3:
            utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
            late_payments = st.number_input("Late Payments (12m)", 0, 20, 0)

        if st.button("Assess Credit Risk", type="primary"):
            debt_to_income = existing_debt / (income + 1)
            payment_history_score = 1 / (1 + late_payments)
            accounts_per_credit_year = num_accounts / (credit_age + 0.5)
            loan_to_income_ratio = loan_amount / (income + 1)
            risk_interaction = utilization * debt_to_income
            credit_age_bucket = 0 if credit_age < 2 else 1 if credit_age < 5 else 2 if credit_age < 10 else 3 if credit_age < 20 else 4

            row = pd.DataFrame([{
                "age": age, "annual_income": income, "employment_years": employment_years,
                "num_accounts": num_accounts, "credit_age_years": credit_age,
                "existing_debt": existing_debt, "loan_amount": loan_amount,
                "credit_utilization": utilization, "late_payments_12m": late_payments,
                "debt_to_income": debt_to_income, "payment_history_score": payment_history_score,
                "debt_to_income_ratio": debt_to_income, "credit_age_bucket": credit_age_bucket,
                "accounts_per_credit_year": accounts_per_credit_year,
                "loan_to_income_ratio": loan_to_income_ratio, "risk_interaction": risk_interaction,
            }])[credit_cols]
            row_scaled = row.copy()
            row_scaled[credit_cols] = credit_scaler.transform(row[credit_cols])

            prob_default = credit_model.predict_proba(row_scaled)[0, 1]
            expected_loss = prob_default * loan_amount * 0.6
            rating = "AAA" if prob_default < 0.05 else "AA" if prob_default < 0.15 else \
                     "A" if prob_default < 0.3 else "BB" if prob_default < 0.5 else "C"

            colA, colB = st.columns([1, 1])
            with colA:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_default * 100,
                    title={"text": "Probability of Default (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkred" if prob_default > 0.5 else "orange" if prob_default > 0.3 else "green"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4f8d4"},
                            {"range": [30, 50], "color": "#fff3cd"},
                            {"range": [50, 100], "color": "#f8d4d4"},
                        ],
                    }
                ))
                gauge.update_layout(height=300, margin=dict(t=50, b=10))
                st.plotly_chart(gauge, use_container_width=True)

            with colB:
                st.metric("Expected Loss (EL)", f"${expected_loss:,.2f}")
                st.metric("Credit Rating", rating)
                if prob_default > 0.5:
                    st.error("⚠️ HIGH RISK — recommend manual review / decline.")
                elif prob_default > 0.3:
                    st.warning("⚠️ Medium risk — consider higher interest rate / collateral.")
                else:
                    st.success("✅ Low risk — approve.")

            fig_pos = px.histogram(credit_df, x="debt_to_income", nbins=40, title="Your Debt-to-Income vs Population")
            fig_pos.add_vline(x=debt_to_income, line_color="red", line_width=3,
                               annotation_text="This applicant")
            st.plotly_chart(fig_pos, use_container_width=True)

    # =======================================================================
    # TAB 3: FRAUD MONITORING
    # =======================================================================
    with tab_fraud:
        st.header("Fraud Monitoring")

        sys.path.append(str(BASE))
        from src.feature_engineering import build_fraud_features
        from src.data_preprocessing import encode_categoricals

        n_sample = st.slider("Number of transactions to monitor", 100, 2000, 500, step=100)
        raw = txn_df.sample(n_sample, random_state=1).copy()

        feat = build_fraud_features(raw)
        feat, _ = encode_categoricals(feat, ["merchant_category"])
        feat_for_model = feat.drop(columns=["transaction_id", "is_fraud"])[fraud_cols]
        feat_scaled = feat_for_model.copy()
        feat_scaled[fraud_cols] = fraud_scaler.transform(feat_for_model[fraud_cols])

        raw["fraud_probability"] = fraud_model.predict_proba(feat_scaled)[:, 1]
        raw["alert"] = raw["fraud_probability"] > 0.5

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Shown", len(raw))
        c2.metric("🚨 High-Risk Alerts", int(raw["alert"].sum()))
        c3.metric("Avg Fraud Probability", f"{raw['fraud_probability'].mean():.2%}")

        col1, col2 = st.columns(2)
        with col1:
            fig_dist = px.histogram(
                raw, x="fraud_probability", nbins=40, title="Fraud Probability Distribution",
                color_discrete_sequence=["#e74c3c"]
            )
            fig_dist.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Alert threshold")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            alerts_by_cat = raw[raw["alert"]].groupby("merchant_category").size().reset_index(name="alerts")
            if len(alerts_by_cat):
                fig_cat = px.bar(alerts_by_cat, x="merchant_category", y="alerts",
                                  title="High-Risk Alerts by Merchant Category",
                                  color="alerts", color_continuous_scale="Reds")
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No high-risk alerts in this sample.")

        fig_scatter = px.scatter(
            raw, x="amount", y="fraud_probability", color="alert",
            hover_data=["merchant_category", "hour_of_day", "distance_from_home_km"],
            title="Transaction Amount vs Fraud Probability",
            color_discrete_map={True: "#e74c3c", False: "#3498db"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Live Transaction Table")
        st.dataframe(
            raw.sort_values("fraud_probability", ascending=False)
               [["transaction_id", "amount", "merchant_category", "hour_of_day",
                 "distance_from_home_km", "is_foreign", "fraud_probability", "alert"]]
               .head(50),
            use_container_width=True
        )

    # =======================================================================
    # TAB 4: MODEL PERFORMANCE
    # =======================================================================
    with tab_perf:
        st.header("Model Performance Comparison")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Credit Risk Models")
            if credit_cmp is not None:
                fig_credit_cmp = px.bar(
                    credit_cmp.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"]),
                    x="model", y="value", color="variable", barmode="group",
                    title="Credit Risk — Metric Comparison Across Models"
                )
                fig_credit_cmp.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_credit_cmp, use_container_width=True)
                st.dataframe(credit_cmp, use_container_width=True)
            else:
                st.info("Run `python main.py` to generate comparison data.")

        with colB:
            st.subheader("Fraud Detection Models")
            if fraud_cmp is not None:
                fig_fraud_cmp = px.bar(
                    fraud_cmp.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"]),
                    x="model", y="value", color="variable", barmode="group",
                    title="Fraud Detection — Metric Comparison Across Models"
                )
                fig_fraud_cmp.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_fraud_cmp, use_container_width=True)
                st.dataframe(fraud_cmp, use_container_width=True)
            else:
                st.info("Run `python main.py` to generate comparison data.")

        st.subheader("ROC Curve & Confusion Matrix (Fraud Model, on sample data)")
        sample = txn_df.sample(min(5000, len(txn_df)), random_state=7).copy()
        feat_s = build_fraud_features(sample)
        feat_s, _ = encode_categoricals(feat_s, ["merchant_category"])
        Xs = feat_s.drop(columns=["transaction_id", "is_fraud"])[fraud_cols]
        Xs_scaled = Xs.copy()
        Xs_scaled[fraud_cols] = fraud_scaler.transform(Xs[fraud_cols])
        y_true = sample["is_fraud"].values
        y_proba = fraud_model.predict_proba(Xs_scaled)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        colC, colD = st.columns(2)
        with colC:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc_val = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc_val:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        with colD:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Legit", "Fraud"], y=["Legit", "Fraud"],
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # =======================================================================
    # TAB 5: FEATURE INSIGHTS
    # =======================================================================
    with tab_features:
        st.header("Feature Importance & Explainability")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Credit Risk — Top Features")
            fi_path = OUT_DIR / "credit_feature_importance.csv"
            if fi_path.exists():
                fi = pd.read_csv(fi_path).head(10)
                fig_fi_credit = px.bar(
                    fi, x="importance", y="feature", orientation="h",
                    title="Top 10 Features Driving Credit Risk", color="importance",
                    color_continuous_scale="Blues"
                )
                fig_fi_credit.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_fi_credit, use_container_width=True)
            else:
                st.info("Run `python main.py` to generate feature importance.")

        with colB:
            st.subheader("Fraud Detection — Top Features")
            fi_path2 = OUT_DIR / "fraud_feature_importance.csv"
            if fi_path2.exists():
                fi2 = pd.read_csv(fi_path2).head(10)
                fig_fi_fraud = px.bar(
                    fi2, x="importance", y="feature", orientation="h",
                    title="Top 10 Features Driving Fraud Detection", color="importance",
                    color_continuous_scale="Reds"
                )
                fig_fi_fraud.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_fi_fraud, use_container_width=True)
            else:
                st.info("Run `python main.py` to generate feature importance.")

        rules_path = OUT_DIR / "credit_business_rules.txt"
        if rules_path.exists():
            with st.expander("📋 View Simplified Business Rules (Decision Tree)"):
                st.code(rules_path.read_text())

st.sidebar.header("About")
st.sidebar.info(
    "This dashboard is powered by models trained in `main.py`. "
    "Retrain anytime by running `python main.py` again, then refresh this page."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tabs:**\n- 📊 Overview\n- 💳 Credit Risk Scoring\n- 🚨 Fraud Monitoring\n- 📈 Model Performance\n- 🔍 Feature Insights")
