"""
ATM Fraud Detection - Feature Engineering & Preprocessing
=========================================================
Handles all data cleaning, encoding and feature extraction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─── Feature Lists ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "pin_failures",
    "velocity_1h",
    "velocity_24h",
    "distance_from_home_km",
    "balance_after",
    "account_age_days",
    "transaction_success",
]
CATEGORICAL_FEATURES = ["atm_location", "card_type"]
TARGET              = "is_fraud"


class ATMPreprocessor:
    """Fit-transform pipeline for ATM fraud data."""

    def __init__(self):
        self.scaler          = StandardScaler()
        self.label_encoders  = {}
        self.feature_columns = []

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on training data and return (X_processed, y)."""
        df = self._clean(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._engineer_features(df)
        X, y = self._split_xy(df)
        X_scaled = self._scale(X, fit=True)
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform unseen data using fitted parameters."""
        df = self._clean(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._engineer_features(df)
        X, y = self._split_xy(df)
        X_scaled = self._scale(X, fit=False)
        return X_scaled, y

    def transform_single(self, record: dict) -> pd.DataFrame:
        """Transform a single transaction record (dict → DataFrame)."""
        df = pd.DataFrame([record])
        df = self._clean(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._engineer_features(df)
        # Drop target if present
        for col in [TARGET, "transaction_id", "timestamp"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        # Align columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        return pd.DataFrame(self.scaler.transform(df),
                            columns=self.feature_columns)

    # ── Internal Steps ────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.drop_duplicates(inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        # Clip extreme outliers
        df["amount"] = df["amount"].clip(upper=10_000)
        df["distance_from_home_km"] = df["distance_from_home_km"].clip(upper=10_000)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived risk-signal features."""
        # Night transaction flag  (10 PM – 5 AM)
        df["is_night"] = df["hour"].apply(
            lambda h: 1 if (h >= 22 or h <= 5) else 0
        )
        # High-amount flag  (above $1000)
        df["is_high_amount"] = (df["amount"] > 1000).astype(int)
        # High velocity flag  (>5 txns in last hour)
        df["is_high_velocity"] = (df["velocity_1h"] > 5).astype(int)
        # Far-from-home flag  (>100 km)
        df["is_far_from_home"] = (df["distance_from_home_km"] > 100).astype(int)
        # PIN risk score  (0–3 capped)
        df["pin_risk"] = df["pin_failures"].clip(upper=3)
        # Amount-to-balance ratio
        df["amount_balance_ratio"] = df.apply(
            lambda r: r["amount"] / (abs(r["balance_after"]) + 1), axis=1
        )
        return df

    def _split_xy(self, df: pd.DataFrame):
        drop_cols = ["transaction_id", "timestamp", TARGET]
        drop_cols = [c for c in drop_cols if c in df.columns]
        y = df[TARGET] if TARGET in df.columns else pd.Series(dtype=int)
        X = df.drop(columns=drop_cols, errors="ignore")
        self.feature_columns = list(X.columns)
        return X, y

    def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            arr = self.scaler.fit_transform(X)
        else:
            arr = self.scaler.transform(X)
        return pd.DataFrame(arr, columns=X.columns)


if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset(1000, 50)
    preprocessor = ATMPreprocessor()
    X, y = preprocessor.fit_transform(df)
    print("Feature columns:", list(X.columns))
    print("X shape:", X.shape, "| y shape:", y.shape)
