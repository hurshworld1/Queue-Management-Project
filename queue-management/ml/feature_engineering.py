import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
import time


class FeatureEngineer:
    """Compute real-time queue features from observations and events."""

    def __init__(self, window_seconds: float = 300):
        self.window_seconds = window_seconds
        self._service_times: deque = deque()
        self._wait_times: deque = deque()
        self._queue_lengths: deque = deque()  # (timestamp, length)
        self._departures: deque = deque()  # timestamps
        self._arrivals: deque = deque()  # timestamps

    def record_service_time(self, service_time_seconds: float, timestamp: float = None):
        ts = timestamp or time.time()
        self._service_times.append((ts, service_time_seconds))
        self._cleanup_old(self._service_times)

    def record_wait_time(self, wait_seconds: float, timestamp: float = None):
        ts = timestamp or time.time()
        self._wait_times.append((ts, wait_seconds))
        self._cleanup_old(self._wait_times)

    def record_queue_length(self, length: int, timestamp: float = None):
        ts = timestamp or time.time()
        self._queue_lengths.append((ts, length))
        self._cleanup_old(self._queue_lengths)

    def record_arrival(self, timestamp: float = None):
        ts = timestamp or time.time()
        self._arrivals.append((ts, 1))  # (timestamp, dummy_value)
        self._cleanup_old(self._arrivals)

    def record_departure(self, timestamp: float = None):
        ts = timestamp or time.time()
        self._departures.append((ts, 1))  # (timestamp, dummy_value)
        self._cleanup_old(self._departures)

    def compute_features(
        self,
        current_queue_length: int,
        active_counters: int = 1,
    ) -> Dict[str, float]:
        """Compute the full feature vector for ML prediction."""
        now = time.time()

        arrival_rate = self._compute_rate(self._arrivals, now, window=60)
        departure_rate = self._compute_rate(self._departures, now, window=60)
        estimated_service_rate = departure_rate if departure_rate > 0 else 0.0

        service_times = [st for _, st in self._service_times]
        mean_service_time = np.mean(service_times) if service_times else 0.0
        median_service_time = np.median(service_times) if service_times else 0.0
        service_time_variability = float(np.std(service_times)) if len(service_times) > 1 else 0.0

        queue_growth_rate = self._compute_queue_growth(now)
        recent_waiting_history = self._compute_avg_wait()

        # Time features
        dt = datetime.fromtimestamp(now)
        time_of_day = dt.hour + dt.minute / 60.0
        day_of_week = dt.weekday()

        # Anomaly flag: no departure in 4 minutes while queue is non-empty
        recent_anomaly_flag = self._check_anomaly(now)

        return {
            "queue_length": current_queue_length,
            "arrival_rate": arrival_rate,
            "departure_rate": departure_rate,
            "estimated_service_rate": estimated_service_rate,
            "mean_service_time": mean_service_time,
            "median_service_time": median_service_time,
            "service_time_variability": service_time_variability,
            "active_counters": active_counters,
            "queue_growth_rate": queue_growth_rate,
            "recent_waiting_history": recent_waiting_history,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "recent_anomaly_flag": float(recent_anomaly_flag),
        }

    def _compute_rate(self, event_queue: deque, now: float, window: float = 60) -> float:
        cutoff = now - window
        count = sum(1 for item in event_queue if item[0] >= cutoff)
        return count / (window / 60.0)  # per minute

    def _compute_queue_growth(self, now: float) -> float:
        cutoff = now - 120  # last 2 minutes
        recent = [(ts, length) for ts, length in self._queue_lengths if ts >= cutoff]
        if len(recent) < 2:
            return 0.0

        old_length = recent[0][1]
        new_length = recent[-1][1]
        dt = (recent[-1][0] - recent[0][0]) / 60.0  # minutes
        if dt <= 0:
            return 0.0
        return (new_length - old_length) / dt

    def _compute_avg_wait(self) -> float:
        waits = [w for _, w in self._wait_times]
        return float(np.mean(waits[-10:])) if waits else 0.0

    def _check_anomaly(self, now: float) -> bool:
        if not self._queue_lengths:
            return False
        current_length = self._queue_lengths[-1][1]
        if current_length == 0:
            return False
        # No departure in 4 minutes
        cutoff = now - 240
        recent_departures = sum(1 for ts in self._departures if ts >= cutoff)
        return recent_departures == 0

    def _cleanup_old(self, queue: deque):
        cutoff = time.time() - self.window_seconds
        while queue and queue[0][0] < cutoff:
            queue.popleft()
