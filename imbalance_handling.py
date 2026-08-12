"""
Phase 4: Feature Engineering
------------------------------
Separate feature builders for the two use-cases:
  - build_credit_features()  -> credit risk features
  - build_fraud_features()   -> fraud detection features
"""

import numpy as np
import pandas as pd


def build_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Credit utilization ratio (already present, keep as-is)
    # Payment history feature
    df["payment_history_score"] = 1 / (1 + df["late_payments_12m"])

    # Debt-to-income ratio (recompute safely)
    df["debt_to_income_ratio"] = df["existing_debt"] / (df["annual_income"] + 1)

    # Credit age buckets
    df["credit_age_bucket"] = pd.cut(
        df["credit_age_years"], bins=[-1, 2, 5, 10, 20, 100],
        labels=["very_new", "new", "established", "mature", "very_mature"]
    ).astype(str)

    # Number of accounts feature: accounts per year of credit history
    df["accounts_per_credit_year"] = df["num_accounts"] / (df["credit_age_years"] + 0.5)

    # Loan-to-income ratio
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1)

    # Interaction feature
    df["risk_interaction"] = df["credit_utilization"] * df["debt_to_income_ratio"]

    return df


def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time-based features
    df["is_night_txn"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h >= 23 else 0)

    # Location anomaly proxy
    df["far_from_home"] = (df["distance_from_home_km"] > 100).astype(int)

    # Transaction velocity flag
    df["high_velocity"] = (df["txn_velocity_1h"] >= 3).astype(int)

    # Amount z-score within merchant category (user/merchant behavior pattern)
    df["amount_zscore_by_category"] = df.groupby("merchant_category")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # Composite anomaly score (simple heuristic feature, models can still learn nonlinear combos)
    df["anomaly_score"] = (
        df["is_night_txn"] + df["far_from_home"] + df["high_velocity"] + df["is_foreign"]
    )

    return df
