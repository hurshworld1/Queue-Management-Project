import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from ml.preprocessing import (
    clean_data, prepare_features, create_time_features, FEATURE_COLUMNS,
)
from ml.feature_engineering import FeatureEngineer
from ml.evaluate import compute_metrics


class TestPreprocessing:
    def _make_sample_df(self):
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1min"),
            "queue_length": [0, 1, 2, 3, 5, 4, 3, 2, 1, 0] * 2,
            "arrival_rate": [1.0, 1.5, 2.0, 1.0, 0.5] * 4,
            "departure_rate": [0.5, 1.0, 1.5, 1.0, 0.5] * 4,
            "active_counters": [1, 1, 2, 2, 1] * 4,
            "mean_service_time": [30, 35, 25, 40, 30] * 4,
            "median_service_time": [28, 33, 23, 38, 28] * 4,
            "queue_growth_rate": [0.5, -0.3, 0.2, 0.1, -0.5] * 4,
            "predicted_wait": [10, 15, 20, 30, 25] * 4,
            "actual_wait": [12, 14, 22, 28, 23] * 4,
            "recent_waiting_history": [10, 12, 15, 20, 18] * 4,
            "time_of_day": [10, 10, 10, 10, 10] * 4,
            "day_of_week": [0] * 20,
            "recent_anomaly_flag": [0] * 20,
            "estimated_service_rate": [0.5, 1.0, 1.5, 1.0, 0.5] * 4,
            "service_time_variability": [5, 3, 8, 2, 4] * 4,
        })

    def test_clean_data(self):
        df = self._make_sample_df()
        df.loc[2, "queue_length"] = -1
        df.loc[4, "actual_wait"] = -5
        cleaned = clean_data(df)
        assert len(cleaned) <= len(df)
        assert (cleaned["actual_wait"] >= 0).all()

    def test_prepare_features(self):
        df = self._make_sample_df()
        X, y = prepare_features(df, FEATURE_COLUMNS[:5])
        assert X.shape[1] == 5
        assert y is not None
        assert len(y) == len(df)

    def test_create_time_features(self):
        df = self._make_sample_df()
        result = create_time_features(df)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "minute_of_day" in result.columns


class TestFeatureEngineer:
    def test_initial_features(self):
        fe = FeatureEngineer()
        features = fe.compute_features(current_queue_length=5, active_counters=2)
        assert features["queue_length"] == 5
        assert features["active_counters"] == 2
        assert features["time_of_day"] >= 0
        assert features["day_of_week"] >= 0

    def test_arrival_rate(self):
        import time
        fe = FeatureEngineer()
        now = time.time()
        for _ in range(5):
            fe.record_arrival(now)
        rate = fe._compute_rate(fe._arrivals, now, window=60)
        assert rate > 0

    def test_service_time_recording(self):
        fe = FeatureEngineer()
        fe.record_service_time(30.0)
        fe.record_service_time(45.0)
        assert len(fe._service_times) == 2


class TestEvaluation:
    def test_compute_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 18.0, 33.0, 38.0, 55.0])
        metrics = compute_metrics(y_true, y_pred)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert metrics["MAE"] > 0
        assert metrics["R2"] <= 1.0

    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        metrics = compute_metrics(y, y)
        assert metrics["MAE"] == 0.0
        assert metrics["R2"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
