import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "median_absolute_error": float(median_ae),
        "mean_absolute_percentage_error": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100
        ),
    }


def compare_models(y_true, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare multiple models on the same test set."""
    results = []
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = name
        results.append(metrics)

    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("MAE")


def plot_predictions_vs_actual(
    y_true, y_pred, model_name: str, save_path: Optional[str] = None
):
    """Plot predicted vs actual waiting times."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    max_val = max(max(y_true), max(y_pred))
    axes[0].plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual Wait (seconds)")
    axes[0].set_ylabel("Predicted Wait (seconds)")
    axes[0].set_title(f"{model_name}: Predicted vs Actual")
    axes[0].legend()

    # Residuals
    residuals = np.array(y_true) - np.array(y_pred)
    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{model_name}: Residual Distribution")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot comparison of multiple models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    comparison_df[["MAE"]].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("MAE (lower is better)")
    axes[0].set_ylabel("Seconds")

    comparison_df[["RMSE"]].plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("RMSE (lower is better)")
    axes[1].set_ylabel("Seconds")

    comparison_df[["R2"]].plot(kind="bar", ax=axes[2], color="forestgreen")
    axes[2].set_title("R² (higher is better)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def evaluate_baseline_vs_ml(
    y_true,
    y_baseline,
    y_ml,
    baseline_name: str = "Linear Baseline",
    ml_name: str = "ML Model",
) -> Dict:
    """Compare baseline against ML model."""
    baseline_metrics = compute_metrics(y_true, y_baseline)
    ml_metrics = compute_metrics(y_true, y_ml)

    improvement = {
        "MAE_reduction_pct": (baseline_metrics["MAE"] - ml_metrics["MAE"]) / baseline_metrics["MAE"] * 100,
        "RMSE_reduction_pct": (baseline_metrics["RMSE"] - ml_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100,
        "R2_improvement": ml_metrics["R2"] - baseline_metrics["R2"],
    }

    return {
        "baseline": {**baseline_metrics, "name": baseline_name},
        "ml_model": {**ml_metrics, "name": ml_name},
        "improvement": improvement,
    }
