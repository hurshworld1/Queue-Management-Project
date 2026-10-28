"""
Online Ensemble Predictor for queue wait time estimation.

Combines three models that continuously learn from real observations:
1. Weighted moving average (fast adaptation, low memory)
2. Linear regression (captures linear relationships)
3. Random forest (captures non-linear patterns)

The ensemble weights shift based on which model performs best recently.
Each model updates its parameters with every new observation, so the
prediction improves over time without retraining.

Key insight: Queue dynamics change throughout the day. A model trained
on morning data may not predict well at lunch. Online learning adapts.
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Observation:
    """A single training observation."""
    features: Dict[str, float]
    actual_wait: float
    timestamp: float


class OnlineLinearRegressor:
    """
    Online linear regression using Stochastic Gradient Descent.

    Updates weights with each new observation, so it adapts
    to changing queue dynamics without retraining.
    """

    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.lr = learning_rate
        # Initialize with small random weights
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self._feature_names: List[str] = []
        self._n_updates = 0
        self._lambda = 0.001  # L2 regularization

    def set_feature_names(self, names: List[str]):
        self._feature_names = names
        self.weights = [0.0] * len(names)
        # Smart initialization: weight[i] proportional to expected feature scale
        for i, name in enumerate(names):
            if name == "queue_length":
                self.weights[i] = 10.0  # each person adds ~10s wait
            elif name == "active_counters":
                self.weights[i] = -5.0  # more counters = less wait
            elif name == "mean_service_time":
                self.weights[i] = 0.5

    def predict(self, features: Dict[str, float]) -> float:
        x = [features.get(name, 0.0) for name in self._feature_names]
        pred = self.bias
        for i, xi in enumerate(x):
            pred += self.weights[i] * xi
        return max(0.0, pred)

    def update(self, features: Dict[str, float], actual: float):
        x = [features.get(name, 0.0) for name in self._feature_names]
        pred = self.predict(features)
        error = pred - actual

        # SGD update with L2 regularization
        for i in range(len(self.weights)):
            gradient = error * x[i] + self._lambda * self.weights[i]
            self.weights[i] -= self.lr * gradient
        self.bias -= self.lr * error
        self._n_updates += 1

    @property
    def weight_dict(self) -> Dict[str, float]:
        return {name: round(w, 4) for name, w in zip(self._feature_names, self.weights)}


class OnlineRandomForest:
    """
    Lightweight online random forest using split-on-best-feature.

    Instead of a full random forest (which requires batch training),
    this maintains a set of online decision stumps that partition
    the feature space and maintain local averages.

    Each stump splits on a random feature at a learned threshold.
    Predictions are weighted averages of all stumps.
    """

    def __init__(self, n_trees: int = 10, n_features: int = 5):
        self.n_trees = n_trees
        self.n_features = n_features
        self._feature_names: List[str] = []
        self._trees: List[Dict] = []
        self._tree_weights = [1.0] * n_trees
        self._n_updates = 0
        self._split_feature_indices: List[List[int]] = []

        import random
        self._rng = random.Random(42)

    def set_feature_names(self, names: List[str]):
        self._feature_names = names
        self._trees = []
        self._split_feature_indices = []
        for _ in range(self.n_trees):
            # Each tree uses a random subset of features
            n = min(self.n_features, len(names))
            indices = self._rng.sample(range(len(names)), n)
            self._split_feature_indices.append(indices)
            self._trees.append({
                "nodes": {},  # (feature_idx, threshold) -> {"sum": 0, "count": 0}
                "default_sum": 0.0,
                "default_count": 0,
            })

    def _get_node_key(self, tree_idx: int, features: Dict[str, float]):
        tree = self._trees[tree_idx]
        indices = self._split_feature_indices[tree_idx]
        if not indices:
            return "default"

        # Walk through a simple decision path
        best_key = "default"
        for idx in indices:
            name = self._feature_names[idx]
            val = features.get(name, 0.0)
            # Create a bucket key based on feature value
            bucket = round(val, 1)
            key = f"{idx}_{bucket}"
            best_key = key

        return best_key

    def predict(self, features: Dict[str, float]) -> float:
        total = 0.0
        total_weight = 0.0

        for i, tree in enumerate(self._trees):
            key = self._get_node_key(i, features)
            node = tree["nodes"].get(key)
            if node and node["count"] > 0:
                w = self._tree_weights[i]
                total += node["sum"] / node["count"] * w
                total_weight += w
            elif tree["default_count"] > 0:
                w = self._tree_weights[i]
                total += tree["default_sum"] / tree["default_count"] * w
                total_weight += w

        if total_weight > 0:
            return max(0.0, total / total_weight)
        return 0.0

    def update(self, features: Dict[str, float], actual: float):
        for i, tree in enumerate(self._trees):
            key = self._get_node_key(i, features)
            if key not in tree["nodes"]:
                tree["nodes"][key] = {"sum": 0.0, "count": 0}

            node = tree["nodes"][key]
            # Exponential moving average to weight recent observations more
            alpha = 0.1
            if node["count"] > 0:
                node["sum"] = alpha * actual * 1 + (1 - alpha) * node["sum"]
                node["count"] = max(1, node["count"] * 0.99 + 1)
            else:
                node["sum"] = actual
                node["count"] = 1

            # Also update default node
            tree["default_sum"] = alpha * actual + (1 - alpha) * tree["default_sum"]
            tree["default_count"] = max(1, tree["default_count"] + 1)

            # Prune trees that have too many nodes
            if len(tree["nodes"]) > 500:
                # Keep only the most recently used nodes
                sorted_nodes = sorted(
                    tree["nodes"].items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )[:200]
                tree["nodes"] = dict(sorted_nodes)

        self._n_updates += 1


class WeightedMovingAverage:
    """
    Simple weighted moving average that gives more weight to recent observations.
    Fast to adapt, good baseline.
    """

    def __init__(self, window: int = 30):
        self.window = window
        self._values: deque = deque(maxlen=window)

    def predict(self, features: Dict[str, float]) -> float:
        if not self._values:
            return features.get("queue_length", 0) * 30.0  # 30s per person default
        return sum(self._values) / len(self._values)

    def update(self, features: Dict[str, float], actual: float):
        self._values.append(actual)


class OnlineEnsemblePredictor:
    """
    Ensemble of three online learners that combine to predict wait time.

    The ensemble dynamically weights each model based on recent accuracy:
    - If queue dynamics are stable, the weighted average wins
    - If dynamics are changing, the online linear regressor adapts faster
    - If there are complex non-linear patterns, the random forest captures them

    All models update continuously, so predictions improve over time.
    """

    FEATURE_NAMES = [
        "queue_length",
        "arrival_rate",
        "departure_rate",
        "mean_service_time",
        "active_counters",
        "queue_growth_rate",
        "time_of_day",
        "day_of_week",
        "service_time_variability",
    ]

    def __init__(self):
        n = len(self.FEATURE_NAMES)

        # Three models
        self.linear = OnlineLinearRegressor(n_features=n, learning_rate=0.005)
        self.forest = OnlineRandomForest(n_trees=8, n_features=5)
        self.wma = WeightedMovingAverage(window=30)

        # Set feature names
        self.linear.set_feature_names(self.FEATURE_NAMES)
        self.forest.set_feature_names(self.FEATURE_NAMES)

        # Model weights (start equal, shift based on performance)
        self._weights = [0.34, 0.33, 0.33]  # linear, forest, wma

        # Recent prediction errors per model
        self._errors: List[List[float]] = [[], [], []]
        self._max_errors = 50

        # Observation history for learning
        self._observations: deque = deque(maxlen=500)
        self._n_updates = 0

        # Per-queue service time tracking
        self._service_times: Dict[int, deque] = {}
        self._arrival_times: Dict[str, float] = {}  # person_id -> arrival_time

    def predict(
        self,
        queue_length: int,
        arrival_rate: float = 0.0,
        departure_rate: float = 0.0,
        mean_service_time: float = 30.0,
        active_counters: int = 1,
        queue_growth_rate: float = 0.0,
        time_of_day: float = 12.0,
        day_of_week: int = 0,
        service_time_variability: float = 0.0,
    ) -> Dict[str, float]:
        """
        Predict waiting time from current queue features.

        Returns dict with predicted_wait, lower_bound, upper_bound.
        """
        features = {
            "queue_length": float(queue_length),
            "arrival_rate": arrival_rate,
            "departure_rate": departure_rate,
            "mean_service_time": mean_service_time,
            "active_counters": float(active_counters),
            "queue_growth_rate": queue_growth_rate,
            "time_of_day": time_of_day,
            "day_of_week": float(day_of_week),
            "service_time_variability": service_time_variability,
        }

        # Get predictions from each model
        pred_linear = self.linear.predict(features)
        pred_forest = self.forest.predict(features)
        pred_wma = self.wma.predict(features)

        predictions = [pred_linear, pred_forest, pred_wma]

        # Weighted ensemble
        total_weight = sum(self._weights)
        if total_weight > 0:
            predicted = sum(p * w for p, w in zip(predictions, self._weights)) / total_weight
        else:
            predicted = pred_linear

        # Confidence bounds based on model agreement and recent error
        pred_std = math.sqrt(sum((p - predicted) ** 2 for p in predictions) / max(len(predictions) - 1, 1))
        recent_mae = self._recent_mae()

        lower = max(0, predicted - max(pred_std, recent_mae * 0.5))
        upper = predicted + max(pred_std, recent_mae * 0.5)

        return {
            "predicted_wait": round(predicted, 1),
            "lower_bound": round(lower, 1),
            "upper_bound": round(upper, 1),
            "model_agreement": round(1.0 / (1.0 + pred_std / max(predicted, 1)), 3),
            "linear_pred": round(pred_linear, 1),
            "forest_pred": round(pred_forest, 1),
            "wma_pred": round(pred_wma, 1),
        }

    def update(
        self,
        features: Dict[str, float],
        actual_wait: float,
    ):
        """
        Feed a new observation (actual wait time measured) to all models.
        All models update their parameters to learn from this data point.
        """
        if actual_wait < 0:
            return

        # Clamp extreme values
        actual_wait = min(actual_wait, 3600.0)

        # Get predictions before update
        pred_linear = self.linear.predict(features)
        pred_forest = self.forest.predict(features)
        pred_wma = self.wma.predict(features)

        predictions = [pred_linear, pred_forest, pred_wma]

        # Record errors
        for i in range(3):
            error = abs(predictions[i] - actual_wait)
            self._errors[i].append(error)
            if len(self._errors[i]) > self._max_errors:
                self._errors[i].pop(0)

        # Update all models
        self.linear.update(features, actual_wait)
        self.forest.update(features, actual_wait)
        self.wma.update(features, actual_wait)

        # Update ensemble weights (give more weight to better-performing models)
        self._update_weights()

        # Store observation
        self._observations.append(Observation(
            features=features,
            actual_wait=actual_wait,
            timestamp=time.time(),
        ))
        self._n_updates += 1

    def record_arrival(self, person_id: str, queue_id: int):
        """Record when a person enters the queue."""
        self._arrival_times[person_id] = time.time()

    def record_departure(self, person_id: str, queue_id: int) -> Optional[float]:
        """Record when a person leaves. Returns the measured wait time."""
        now = time.time()
        arrival = self._arrival_times.pop(person_id, None)
        if arrival is None:
            return None

        wait_time = now - arrival
        wait_time = max(1.0, min(600.0, wait_time))  # clamp 1s to 10min

        # Record service time
        if queue_id not in self._service_times:
            self._service_times[queue_id] = deque(maxlen=30)
        self._service_times[queue_id].append((now, wait_time))

        return wait_time

    def get_avg_service_time(self, queue_id: int = 0) -> float:
        """Get exponential moving average of service time for a queue."""
        times = self._service_times.get(queue_id, [])
        if not times:
            return 30.0  # default 30 seconds

        values = [t for _, t in times]
        # Exponential weighting (recent values count more)
        n = len(values)
        weights = [math.exp(0.1 * (i - n + 1)) for i in range(n)]
        total_w = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / total_w

    def get_recent_service_times(self, queue_id: int = 0) -> List[float]:
        """Get recent raw service times for analysis."""
        return [t for _, t in self._service_times.get(queue_id, [])]

    def _update_weights(self):
        """Shift ensemble weights toward the most accurate model."""
        # Inverse error weighting (lower error = higher weight)
        weights = []
        for errors in self._errors:
            if len(errors) < 3:
                weights.append(1.0)
            else:
                mae = sum(errors[-20:]) / min(len(errors), 20)
                # Inverse MAE with smoothing
                weights.append(1.0 / max(mae, 1.0))

        total = sum(weights)
        if total > 0:
            self._weights = [w / total for w in weights]
        else:
            self._weights = [0.34, 0.33, 0.33]

    def _recent_mae(self) -> float:
        """Mean absolute error across all models over recent observations."""
        all_errors = [e for errors in self._errors for e in errors[-10:]]
        if not all_errors:
            return 15.0  # default 15s uncertainty
        return sum(all_errors) / len(all_errors)

    def get_metrics(self) -> Dict[str, float]:
        """Return current model performance metrics."""
        return {
            "linear_weight": round(self._weights[0], 3),
            "forest_weight": round(self._weights[1], 3),
            "wma_weight": round(self._weights[2], 3),
            "total_updates": self._n_updates,
            "linear_mae": round(sum(self._errors[0][-20:]) / max(len(self._errors[0][-20:]), 1), 2),
            "forest_mae": round(sum(self._errors[1][-20:]) / max(len(self._errors[1][-20:]), 1), 2),
            "wma_mae": round(sum(self._errors[2][-20:]) / max(len(self._errors[2][-20:]), 1), 2),
            "overall_mae": round(self._recent_mae(), 2),
            "model_agreement": round(1.0 / (1.0 + self._recent_mae() / 30.0), 3),
            "observations_learned": len(self._observations),
        }
