import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_queue_data(filepath: str) -> pd.DataFrame:
    """Load raw queue observation data."""
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw queue data: remove NaN, fix types, handle outliers."""
    # Drop rows with no queue length (invalid observations)
    df = df.dropna(subset=["queue_length"])

    # Ensure numeric columns
    numeric_cols = [
        "queue_length", "arrival_rate", "departure_rate",
        "active_counters", "mean_service_time", "median_service_time",
        "queue_growth_rate", "predicted_wait", "actual_wait",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove obvious outliers (negative wait times, unreasonable service times)
    if "actual_wait" in df.columns:
        df = df[df["actual_wait"] >= 0]
        df = df[df["actual_wait"] < 3600]  # max 1 hour

    # Fill remaining NaN values
    df = df.fillna(0)

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepare feature matrix and target variable."""
    X = df[feature_columns].copy()

    y = None
    if "actual_wait" in df.columns:
        y = df["actual_wait"]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple:
    """Split data into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val_size, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from timestamp column."""
    if "timestamp" not in df.columns:
        return df

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def normalize_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Normalize features using training set statistics."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, {"scaler": scaler}


FEATURE_COLUMNS = [
    "queue_length",
    "arrival_rate",
    "departure_rate",
    "estimated_service_rate",
    "mean_service_time",
    "median_service_time",
    "service_time_variability",
    "active_counters",
    "queue_growth_rate",
    "recent_waiting_history",
    "time_of_day",
    "day_of_week",
    "recent_anomaly_flag",
]
