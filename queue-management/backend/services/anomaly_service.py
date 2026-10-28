from typing import List, Dict, Optional
from collections import deque
import time
import numpy as np
from dataclasses import dataclass


@dataclass
class AnomalyAlert:
    queue_id: int
    type: str
    severity: str
    evidence: str
    timestamp: float


class AnomalyDetector:
    """Detect unusual service conditions using rolling statistics."""

    def __init__(
        self,
        normal_interval_low: float = 20.0,
        normal_interval_high: float = 60.0,
        no_service_threshold: float = 240.0,  # 4 minutes
        queue_growth_threshold: float = 5.0,
    ):
        self.normal_interval_low = normal_interval_low
        self.normal_interval_high = normal_interval_high
        self.no_service_threshold = no_service_threshold
        self.queue_growth_threshold = queue_growth_threshold

        self._service_completions: Dict[int, deque] = {}
        self._queue_lengths: Dict[int, deque] = {}
        self._last_departure: Dict[int, float] = {}

    def record_service_completion(self, queue_id: int, timestamp: float = None):
        ts = timestamp or time.time()
        if queue_id not in self._service_completions:
            self._service_completions[queue_id] = deque(maxlen=100)
        self._service_completions[queue_id].append(ts)
        self._last_departure[queue_id] = ts

    def record_queue_length(self, queue_id: int, length: int, timestamp: float = None):
        ts = timestamp or time.time()
        if queue_id not in self._queue_lengths:
            self._queue_lengths[queue_id] = deque(maxlen=100)
        self._queue_lengths[queue_id].append((ts, length))

    def check_anomalies(self, queue_id: int, current_queue_length: int) -> List[AnomalyAlert]:
        """Check for anomalous conditions and return alerts."""
        alerts = []
        now = time.time()

        # Check 1: No service completion for too long
        alert = self._check_no_service(queue_id, now, current_queue_length)
        if alert:
            alerts.append(alert)

        # Check 2: Sudden queue growth
        alert = self._check_sudden_growth(queue_id, now)
        if alert:
            alerts.append(alert)

        # Check 3: Service much slower than normal
        alert = self._check_slow_service(queue_id)
        if alert:
            alerts.append(alert)

        return alerts

    def _check_no_service(
        self, queue_id: int, now: float, queue_length: int
    ) -> Optional[AnomalyAlert]:
        if queue_length == 0:
            return None

        last_dep = self._last_departure.get(queue_id)
        if last_dep is None:
            # Never seen a departure - check if queue has been non-empty long enough
            return None

        elapsed = now - last_dep
        if elapsed > self.no_service_threshold:
            normal_interval = self._get_normal_interval(queue_id)
            return AnomalyAlert(
                queue_id=queue_id,
                type="no_service_completion",
                severity="high" if elapsed > self.no_service_threshold * 1.5 else "medium",
                evidence=(
                    f"No detected service completion for {int(elapsed)} seconds, "
                    f"compared with a recent normal interval of approximately "
                    f"{int(normal_interval)} seconds."
                ),
                timestamp=now,
            )
        return None

    def _check_sudden_growth(self, queue_id: int, now: float) -> Optional[AnomalyAlert]:
        lengths = self._queue_lengths.get(queue_id)
        if not lengths or len(lengths) < 2:
            return None

        # Check growth in last 2 minutes
        cutoff = now - 120
        recent = [(ts, length) for ts, length in lengths if ts >= cutoff]
        if len(recent) < 2:
            return None

        growth = recent[-1][1] - recent[0][1]
        if growth >= self.queue_growth_threshold:
            return AnomalyAlert(
                queue_id=queue_id,
                type="sudden_queue_growth",
                severity="medium",
                evidence=f"Queue grew by {growth} people in the last {int(now - recent[0][0])} seconds.",
                timestamp=now,
            )
        return None

    def _check_slow_service(self, queue_id: int) -> Optional[AnomalyAlert]:
        completions = self._service_completions.get(queue_id)
        if not completions or len(completions) < 3:
            return None

        recent = list(completions)[-10:]
        intervals = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]

        normal_low, normal_high = self.normal_interval_low, self.normal_interval_high
        avg_interval = np.mean(intervals)

        if avg_interval > normal_high * 2:
            return AnomalyAlert(
                queue_id=queue_id,
                type="slow_service",
                severity="high",
                evidence=(
                    f"Service is significantly slower than normal: "
                    f"average interval {avg_interval:.0f}s vs normal {normal_high:.0f}s."
                ),
                timestamp=time.time(),
            )
        return None

    def _get_normal_interval(self, queue_id: int) -> float:
        completions = self._service_completions.get(queue_id)
        if not completions or len(completions) < 2:
            return (self.normal_interval_low + self.normal_interval_high) / 2

        recent = list(completions)[-20:]
        intervals = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]
        return float(np.median(intervals))


# Singleton
anomaly_detector = AnomalyDetector()
