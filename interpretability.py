"""
Phase 8: Model Evaluation
-----------------------------
Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
and a simple cost-sensitive evaluation (business cost of FP vs FN).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def cost_sensitive_evaluation(y_test, y_pred, cost_fp=10, cost_fn=100):
    """
    Example: missing a real fraud/default (False Negative) is far costlier
    than flagging a good customer for review (False Positive).
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    return {
        "false_positives": int(fp), "false_negatives": int(fn),
        "total_business_cost": float(total_cost)
    }


def compare_models(results: list) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
