"""
Phase 9: Model Interpretability & Explainability
----------------------------------------------------
Feature importance, SHAP values, and simple business-rule extraction
from a shallow decision tree (easy for non-technical stakeholders).
"""

import numpy as np
import pandas as pd


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model has no feature_importances_ or coef_ attribute")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_with_shap(model, X_sample, feature_names=None):
    """Returns a SHAP Explainer + shap_values for tree-based models."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def extract_business_rules(X_train, y_train, feature_names, max_depth=3):
    """Train a small, human-readable decision tree to approximate the model's logic."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)
    rules = export_text(tree, feature_names=list(feature_names))
    return rules
