"""
Phase 2: Data Preprocessing
----------------------------
- Handle missing values
- Detect / treat outliers (IQR capping)
- Encode categorical columns
- Scale numeric features
- Train/test split helper
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def assess_data_quality(df: pd.DataFrame, name: str = "dataset"):
    print(f"\n--- Data Quality Report: {name} ---")
    print(f"Shape: {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values:\n", missing)
    else:
        print("No missing values.")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    return {"shape": df.shape, "missing": missing.to_dict(), "duplicates": int(df.duplicated().sum())}


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers instead of dropping rows (keeps signal for fraud detection)."""
    df = df.copy()
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def encode_categoricals(df: pd.DataFrame, columns) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_features(X_train, X_test, columns):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test, scaler


def split_data(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42, stratify=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
