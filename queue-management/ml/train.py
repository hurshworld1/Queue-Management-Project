import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from ml.preprocessing import (
    load_queue_data, clean_data, prepare_features, split_data,
    normalize_features, FEATURE_COLUMNS,
)


MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_baseline_linear(X_train, y_train):
    """Train a simple linear regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """Train a Random Forest Regressor."""
    defaults = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(kwargs)
    model = RandomForestRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, **kwargs):
    """Train a Gradient Boosting Regressor."""
    defaults = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_samples_split": 5,
        "random_state": 42,
    }
    defaults.update(kwargs)
    model = GradientBoostingRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """Train an XGBoost Regressor."""
    defaults = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(kwargs)
    model = xgb.XGBRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    median_ae = np.median(np.abs(y_test - y_pred))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "median_absolute_error": float(median_ae),
    }


def train_and_evaluate_all(
    data_path: str, feature_columns: list = None
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    """Train all candidate models and evaluate them."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    # Load and prepare data
    df = load_queue_data(data_path)
    df = clean_data(df)
    X, y = prepare_features(df, feature_columns)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_s, X_val_s, X_test_s, scaler_info = normalize_features(X_train, X_val, X_test)

    # Train all models
    models = {
        "linear_baseline": train_baseline_linear(X_train_s, y_train),
        "random_forest": train_random_forest(X_train_s, y_train),
        "gradient_boosting": train_gradient_boosting(X_train_s, y_train),
        "xgboost": train_xgboost(X_train_s, y_train),
    }

    # Evaluate all
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test_s, y_test)
        print(f"{name}: MAE={results[name]['MAE']:.2f}s, RMSE={results[name]['RMSE']:.2f}s, R2={results[name]['R2']:.4f}")

    # Save best model
    best_name = min(results, key=lambda k: results[k]["MAE"])
    print(f"\nBest model: {best_name}")
    save_model(models[best_name], best_name, results[best_name], scaler_info)

    return models, results


def save_model(model, name: str, metrics: dict, scaler_info: dict = None):
    """Save a trained model to disk."""
    ensure_model_dir()
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODEL_DIR, f"{name}_{version}")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    if scaler_info and "scaler" in scaler_info:
        joblib.dump(scaler_info["scaler"], os.path.join(model_dir, "scaler.joblib"))

    meta = {
        "model_name": name,
        "version": version,
        "metrics": metrics,
        "train_date": datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_dir}")


def load_latest_model(model_name: str = None) -> Tuple[object, Optional[object], dict]:
    """Load the latest saved model."""
    ensure_model_dir()
    dirs = sorted(
        [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))],
        reverse=True,
    )

    if model_name:
        dirs = [d for d in dirs if d.startswith(model_name)]

    if not dirs:
        raise FileNotFoundError(f"No saved model found for '{model_name}'")

    model_dir = os.path.join(MODEL_DIR, dirs[0])
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    scaler = None
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    meta_path = os.path.join(model_dir, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return model, scaler, metadata


def predict_wait(model, scaler, features: Dict[str, float]) -> Dict[str, float]:
    """Predict waiting time from features."""
    import pandas as pd
    feature_df = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    if scaler:
        feature_df = pd.DataFrame(
            scaler.transform(feature_df),
            columns=FEATURE_COLUMNS,
        )
    predicted = model.predict(feature_df)[0]
    return {
        "predicted_wait": float(max(0, predicted)),
        "lower_bound": float(max(0, predicted * 0.8)),
        "upper_bound": float(predicted * 1.2),
    }
