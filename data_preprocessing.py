"""
Phase 7: Ensemble Methods
----------------------------
Voting, Stacking, and Bagging built on top of the base models.
"""

from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_voting_ensemble(models: dict, voting="soft"):
    """models: dict of {name: fitted_or_unfitted_estimator}."""
    estimators = [(name, m) for name, m in models.items()]
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


def build_stacking_ensemble(models: dict, final_estimator=None):
    estimators = [(name, m) for name, m in models.items()]
    final_estimator = final_estimator or LogisticRegression(max_iter=1000)
    return StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)


def build_bagging_ensemble(base_estimator=None, n_estimators=50, random_state=42):
    base_estimator = base_estimator or DecisionTreeClassifier(max_depth=8)
    return BaggingClassifier(
        estimator=base_estimator, n_estimators=n_estimators,
        random_state=random_state, n_jobs=-1
    )
