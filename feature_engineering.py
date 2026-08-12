"""
Phase 6: Classification Models
---------------------------------
Wraps Logistic Regression, Random Forest, XGBoost, LightGBM,
and a small Neural Network (MLP) behind one simple interface.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb


def get_models(class_weight_dict=None, random_state=42):
    """Return a dict of {name: unfitted model}."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight=class_weight_dict, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight=class_weight_dict,
            random_state=random_state, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            eval_metric="logloss", random_state=random_state, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            random_state=random_state, n_jobs=-1, verbose=-1
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=random_state
        ),
    }
    return models


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
