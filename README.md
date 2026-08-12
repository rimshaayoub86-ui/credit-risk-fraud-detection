# Credit Risk Assessment & Fraud Detection System

End-to-end Machine Learning project for financial institutions covering
**credit risk scoring** (probability of default, expected loss, credit rating)
and **fraud detection** (real-time transaction scoring on imbalanced data).

## 📁 Project Structure
```
credit_risk_fraud_detection/
├── data/
│   ├── generate_data.py       # creates synthetic credit_data.csv & transactions.csv
│   ├── credit_data.csv
│   └── transactions.csv
├── src/
│   ├── data_preprocessing.py  # missing values, outliers, encoding, scaling
│   ├── feature_engineering.py # credit + fraud specific features
│   ├── imbalance_handling.py  # SMOTE, ADASYN, undersampling, class weights
│   ├── models.py              # Logistic Regression, RF, XGBoost, LightGBM, MLP
│   ├── ensemble.py            # Voting, Stacking, Bagging
│   ├── evaluation.py          # metrics, confusion matrix, cost-sensitive eval
│   └── interpretability.py    # feature importance, SHAP, business rules
├── dashboard/
│   ├── app.py                 # Streamlit dashboard (credit scoring + fraud monitor)
│   └── api.py                 # Flask scoring API
├── outputs/                   # generated reports, plots, CSVs (after running main.py)
├── models_saved/              # trained model .pkl files (after running main.py)
├── main.py                    # runs the FULL pipeline end-to-end
├── requirements.txt
└── README.md
```

## 🚀 Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate synthetic data
python data/generate_data.py

# 3. Run the full training pipeline (both credit risk + fraud models)
python main.py

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py

# 5. Or launch the scoring API
python dashboard/api.py
```

## 🧠 What main.py does, step by step
1. **Load data** → `data/credit_data.csv` and `data/transactions.csv`
2. **Data quality check** → shape, missing values, duplicates
3. **Clean data** → fill missing values, cap outliers (IQR method)
4. **Feature engineering**
   - Credit: debt-to-income ratio, payment history score, credit age buckets, loan-to-income ratio
   - Fraud: night-transaction flag, distance-from-home anomaly, transaction velocity, amount z-score per merchant category
5. **Encode categoricals**, **scale numeric features**, **train/test split**
6. **Handle imbalance** (fraud only) → SMOTE oversampling + class weights
7. **Train 5 models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network (MLP)
8. **Build ensembles**: Voting (credit) and Stacking (fraud)
9. **Evaluate**: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, cost-sensitive business cost
10. **Interpretability**: feature importance chart + a shallow decision-tree "business rules" text file
11. **Save** best model + scaler + feature list to `models_saved/` for the dashboard/API to reuse

## 📊 Sample results (synthetic data, will vary on real data)
- Fraud detection: XGBoost/Stacking reach ~99.9% ROC-AUC, ~95-96% recall on fraud class
- Credit risk: Logistic Regression baseline ~0.68 ROC-AUC (synthetic signal is intentionally noisy — swap in real data like the Kaggle "Give Me Some Credit" dataset for stronger results)

## 🔁 Using your OWN real data
Replace `data/credit_data.csv` / `data/transactions.csv` with your real files,
keeping the same column names (or update `src/feature_engineering.py` and
`main.py` column references to match your schema). Everything downstream
works unchanged.

## 📌 Notes
- Synthetic data is used here so the project **runs immediately out of the box**
  without needing external downloads or credentials.
- SHAP explainability is implemented in `src/interpretability.py::explain_with_shap`
  — call it manually in a notebook for deep-dive plots (not run in `main.py` by
  default to keep the pipeline fast).
