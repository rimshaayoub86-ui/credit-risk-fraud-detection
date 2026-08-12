"""
MAIN PIPELINE
=================
Runs the full workflow for BOTH sub-systems:
    1. Credit Risk Assessment  (binary classification: default / no default)
    2. Fraud Detection          (binary classification: fraud / legit, imbalanced)

Usage:
    python main.py
Outputs land in ./outputs/ (metrics CSVs, plots, saved models).
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_preprocessing import (
    assess_data_quality, handle_missing_values, cap_outliers_iqr,
    encode_categoricals, scale_features, split_data
)
from src.feature_engineering import build_credit_features, build_fraud_features
from src.imbalance_handling import apply_smote, get_class_weights
from src.models import get_models, train_model
from src.ensemble import build_voting_ensemble, build_stacking_ensemble
from src.evaluation import evaluate_model, cost_sensitive_evaluation, compare_models
from src.interpretability import get_feature_importance, extract_business_rules

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
MODELS_DIR = BASE / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)


def run_credit_risk_pipeline():
    print("\n" + "=" * 60)
    print("CREDIT RISK ASSESSMENT PIPELINE")
    print("=" * 60)

    df = pd.read_csv(BASE / "data" / "credit_data.csv")
    assess_data_quality(df, "credit_data")

    df = handle_missing_values(df)
    num_cols = ["annual_income", "existing_debt", "loan_amount"]
    df = cap_outliers_iqr(df, num_cols)
    df = build_credit_features(df)

    cat_cols = ["credit_age_bucket"]
    df, _ = encode_categoricals(df, cat_cols)
    df = df.drop(columns=["applicant_id"])

    X_train, X_test, y_train, y_test = split_data(df, target_col="default")

    scale_cols = [c for c in X_train.columns]
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, scale_cols)

    class_weights = get_class_weights(y_train)
    models = get_models(class_weight_dict=class_weights)

    results, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name} ...")
        m = train_model(model, X_train_s, y_train)
        metrics, cm = evaluate_model(m, X_test_s, y_test, name)
        results.append(metrics)
        fitted[name] = m

    # Voting ensemble on top of the 3 strongest tree/linear models
    voting = build_voting_ensemble({
        "rf": fitted["RandomForest"], "xgb": fitted["XGBoost"], "lgbm": fitted["LightGBM"]
    })
    voting.fit(X_train_s, y_train)
    v_metrics, v_cm = evaluate_model(voting, X_test_s, y_test, "VotingEnsemble")
    results.append(v_metrics)
    fitted["VotingEnsemble"] = voting

    results_df = compare_models(results)
    print("\nCredit Risk Model Comparison:\n", results_df)
    results_df.to_csv(OUT / "credit_model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest credit risk model: {best_name}")

    # Interpretability
    if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
        fi = get_feature_importance(best_model, X_train_s.columns)
        fi.to_csv(OUT / "credit_feature_importance.csv", index=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=fi.head(10), x="importance", y="feature")
        plt.title(f"Top 10 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(OUT / "credit_feature_importance.png", dpi=120)
        plt.close()

    rules = extract_business_rules(X_train_s, y_train, X_train_s.columns)
    (OUT / "credit_business_rules.txt").write_text(rules)

    # Save artifacts
    joblib.dump(best_model, MODELS_DIR / "credit_risk_best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "credit_scaler.pkl")
    joblib.dump(list(X_train_s.columns), MODELS_DIR / "credit_feature_columns.pkl")

    return results_df


def run_fraud_detection_pipeline():
    print("\n" + "=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 60)

    df = pd.read_csv(BASE / "data" / "transactions.csv")
    assess_data_quality(df, "transactions")

    df = handle_missing_values(df)
    df = build_fraud_features(df)

    cat_cols = ["merchant_category"]
    df, _ = encode_categoricals(df, cat_cols)
    df = df.drop(columns=["transaction_id"])

    X_train, X_test, y_train, y_test = split_data(df, target_col="is_fraud")

    scale_cols = [c for c in X_train.columns]
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, scale_cols)

    print(f"Original train class balance: {y_train.value_counts(normalize=True).to_dict()}")
    X_train_res, y_train_res = apply_smote(X_train_s, y_train)
    print(f"After SMOTE: {pd.Series(y_train_res).value_counts(normalize=True).to_dict()}")

    class_weights = get_class_weights(y_train)
    models = get_models(class_weight_dict=class_weights)

    results, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name} ...")
        m = train_model(model, X_train_res, y_train_res)
        metrics, cm = evaluate_model(m, X_test_s, y_test, name)
        results.append(metrics)
        fitted[name] = m

    stacking = build_stacking_ensemble({
        "rf": fitted["RandomForest"], "xgb": fitted["XGBoost"], "lgbm": fitted["LightGBM"]
    })
    stacking.fit(X_train_res, y_train_res)
    s_metrics, s_cm = evaluate_model(stacking, X_test_s, y_test, "StackingEnsemble")
    results.append(s_metrics)
    fitted["StackingEnsemble"] = stacking

    results_df = compare_models(results)
    print("\nFraud Detection Model Comparison:\n", results_df)
    results_df.to_csv(OUT / "fraud_model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest fraud detection model: {best_name}")

    y_pred_best = best_model.predict(X_test_s)
    cost = cost_sensitive_evaluation(y_test, y_pred_best)
    print("Cost-sensitive evaluation:", cost)
    pd.DataFrame([cost]).to_csv(OUT / "fraud_cost_evaluation.csv", index=False)

    if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
        fi = get_feature_importance(best_model, X_train_s.columns)
        fi.to_csv(OUT / "fraud_feature_importance.csv", index=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=fi.head(10), x="importance", y="feature")
        plt.title(f"Top 10 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(OUT / "fraud_feature_importance.png", dpi=120)
        plt.close()

    joblib.dump(best_model, MODELS_DIR / "fraud_best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "fraud_scaler.pkl")
    joblib.dump(list(X_train_s.columns), MODELS_DIR / "fraud_feature_columns.pkl")

    return results_df


if __name__ == "__main__":
    credit_results = run_credit_risk_pipeline()
    fraud_results = run_fraud_detection_pipeline()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. See ./outputs/ for reports and ./models_saved/ for models.")
    print("=" * 60)
