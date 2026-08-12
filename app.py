"""
Phase 10: Model Deployment - Scoring API
============================================
Simple Flask API exposing two endpoints:
    POST /score/credit   -> returns probability of default + rating
    POST /score/fraud    -> returns fraud probability + alert flag

Run with:
    python dashboard/api.py
Then POST JSON to http://localhost:5000/score/credit  or  /score/fraud
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models_saved"

app = Flask(__name__)

credit_model = joblib.load(MODELS_DIR / "credit_risk_best_model.pkl")
credit_scaler = joblib.load(MODELS_DIR / "credit_scaler.pkl")
credit_cols = joblib.load(MODELS_DIR / "credit_feature_columns.pkl")

fraud_model = joblib.load(MODELS_DIR / "fraud_best_model.pkl")
fraud_scaler = joblib.load(MODELS_DIR / "fraud_scaler.pkl")
fraud_cols = joblib.load(MODELS_DIR / "fraud_feature_columns.pkl")


@app.route("/score/credit", methods=["POST"])
def score_credit():
    payload = request.get_json()
    row = pd.DataFrame([payload])[credit_cols]
    row_scaled = row.copy()
    row_scaled[credit_cols] = credit_scaler.transform(row[credit_cols])
    prob = float(credit_model.predict_proba(row_scaled)[0, 1])
    rating = "AAA" if prob < 0.05 else "AA" if prob < 0.15 else "A" if prob < 0.3 else "BB" if prob < 0.5 else "C"
    return jsonify({"probability_of_default": prob, "credit_rating": rating})


@app.route("/score/fraud", methods=["POST"])
def score_fraud():
    payload = request.get_json()
    row = pd.DataFrame([payload])[fraud_cols]
    row_scaled = row.copy()
    row_scaled[fraud_cols] = fraud_scaler.transform(row[fraud_cols])
    prob = float(fraud_model.predict_proba(row_scaled)[0, 1])
    return jsonify({"fraud_probability": prob, "alert": prob > 0.5})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
