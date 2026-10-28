"""
Queue service with Kalman-smoothed estimates and online ML ensemble prediction.

Architecture:
1. KalmanFilter1D smooths the raw camera queue count (removes jitter)
2. OnlineEnsemblePredictor learns from every observed service completion
3. ServiceTimeTracker records per-counter actual service times with EMA
4. All three combine to produce smooth, accurate, continuously-improving estimates

Key improvement: The wait time estimate doesn't just use a static formula.
It uses an online ML model that learns from real observations. The more people
that go through the queue, the more accurate the predictions become.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import time
import math
from collections import deque

from backend.database.models import (
    Queue, Counter, QueueObservation, Prediction, Anomaly,
    WaitingEvent, ServiceEvent,
)
from ml.kalman_filter import KalmanFilter1D, AdaptiveKalmanFilter, ExponentialMovingAverage
from ml.online_predictor import OnlineEnsemblePredictor


class ServiceTimeTracker:
    """
    Tracks actual service times per counter using Exponential Moving Average.

    The EMA gives more weight to recent service times, so the estimate
    adapts as the counter gets faster or slower throughout the day.
    """

    def __init__(self, half_life: float = 120.0):
        """
        Args:
            half_life: Seconds before old service times lose half their weight.
                       120s = adapts to changes within 2 minutes.
        """
        self._half_life = half_life
        # Per-counter EMAs: {counter_id: EMA}
        self._counter_ema: Dict[int, ExponentialMovingAverage] = {}
        # Global EMA
        self.global_ema = ExponentialMovingAverage(half_life=half_life)
        # Raw recent times for analysis
        self._recent_times: deque = deque(maxlen=50)
        # Per-counter raw times
        self._counter_times: Dict[int, deque] = {}
        # Departure tracking for rate calculation
        self._departures: deque = deque(maxlen=200)
        # Arrival tracking
        self._arrivals: Dict[str, tuple] = {}

    def record_arrival(self, person_id: str, queue_id: int):
        self._arrivals[person_id] = (time.time(), queue_id)

    def record_service_completion(self, person_id: str, counter_id: int = None) -> Optional[float]:
        """
        Record that a person finished being served.

        Returns the measured service time in seconds, or None if
        arrival time wasn't recorded.
        """
        now = time.time()
        self._departures.append(now)

        if person_id in self._arrivals:
            arrival_ts, queue_id = self._arrivals.pop(person_id)
            service_time = now - arrival_ts
            service_time = max(2.0, min(600.0, service_time))
            self._record_time(service_time, counter_id)
            return service_time
        return None

    def record_service_time_direct(self, service_time: float, counter_id: int = None):
        """Record a known service time directly (e.g., from API)."""
        now = time.time()
        self._departures.append(now)
        service_time = max(2.0, min(600.0, service_time))
        self._record_time(service_time, counter_id)

    def _record_time(self, service_time: float, counter_id: int = None):
        now = time.time()
        self._recent_times.append((now, service_time))

        # Update global EMA
        self.global_ema.update(service_time, now)

        # Update per-counter EMA
        cid = counter_id or 0
        if cid not in self._counter_ema:
            self._counter_ema[cid] = ExponentialMovingAverage(half_life=self._half_life)
            self._counter_times[cid] = deque(maxlen=30)
        self._counter_ema[cid].update(service_time, now)
        self._counter_times[cid].append((now, service_time))

    def get_avg_service_time(self) -> float:
        """Get smoothed average service time across all counters."""
        return self.global_ema.value

    def get_avg_per_counter(self) -> Dict[int, float]:
        """Get smoothed average service time per counter."""
        return {cid: ema.value for cid, ema in self._counter_ema.items()}

    def get_recent_times(self, last_n: int = None) -> List[float]:
        """Get most recent raw service times."""
        times = [t for _, t in self._recent_times]
        if last_n:
            return times[-last_n:]
        return times

    def get_departure_rate(self, window_seconds: float = 60) -> float:
        """Departures per minute in the last window."""
        now = time.time()
        count = sum(1 for t in self._departures if now - t < window_seconds)
        return count

    def get_service_time_variability(self) -> float:
        """Standard deviation of recent service times."""
        times = self.get_recent_times(20)
        if len(times) < 3:
            return 5.0  # default low variability
        mean = sum(times) / len(times)
        var = sum((t - mean) ** 2 for t in times) / len(times)
        return math.sqrt(var)


class QueueStateManager:
    """
    In-memory state manager with Kalman-smoothed queue counts
    and online ML ensemble wait time prediction.
    """

    def __init__(self):
        self.queue_states: Dict[int, dict] = {}
        self.service_trackers: Dict[int, ServiceTimeTracker] = {}
        self.kalman_filters: Dict[int, KalmanFilter1D] = {}
        # One shared online predictor that learns across all queues
        self.predictor = OnlineEnsemblePredictor()
        # Per-queue arrival tracking for growth rate
        self._queue_lengths: Dict[int, deque] = {}
        # Arrival/departure timestamps for rate calculation
        self._arrivals: Dict[int, deque] = {}
        self._departures: Dict[int, deque] = {}

    def get_or_create_state(self, queue_id: int) -> dict:
        if queue_id not in self.queue_states:
            self.queue_states[queue_id] = {
                "queue_id": queue_id,
                "queue_length": 0,
                "raw_queue_length": 0,
                "active_counters": 1,
                "estimated_wait_seconds": 0.0,
                "trend": "stable",
                "status": "low",
                "last_updated": datetime.now(timezone.utc),
                "lower_bound": None,
                "upper_bound": None,
                "avg_service_time": 30.0,
                "model_agreement": 0.0,
            }
        if queue_id not in self.service_trackers:
            self.service_trackers[queue_id] = ServiceTimeTracker()
        if queue_id not in self.kalman_filters:
            # Adaptive Kalman filter tuned for queue count smoothing
            self.kalman_filters[queue_id] = AdaptiveKalmanFilter(
                process_noise=0.3,      # queue count can change by ~1 per update
                measurement_noise=1.5,   # camera detection is noisy
                dt=0.2,                  # expected 200ms between updates
            )
        if queue_id not in self._queue_lengths:
            self._queue_lengths[queue_id] = deque(maxlen=100)
        if queue_id not in self._arrivals:
            self._arrivals[queue_id] = deque(maxlen=200)
        if queue_id not in self._departures:
            self._departures[queue_id] = deque(maxlen=200)
        return self.queue_states[queue_id]

    def get_tracker(self, queue_id: int) -> ServiceTimeTracker:
        self.get_or_create_state(queue_id)
        return self.service_trackers[queue_id]

    def update_queue_length(self, queue_id: int, raw_length: int):
        """
        Update queue length with Kalman smoothing.

        The raw camera count is noisy (people detected in one frame
        but not the next). The Kalman filter smooths this to produce
        a stable, responsive count.
        """
        state = self.get_or_create_state(queue_id)
        kalman = self.kalman_filters[queue_id]

        # Store raw value
        state["raw_queue_length"] = raw_length

        # Apply Kalman filter
        smoothed = kalman.update(raw_length)
        smoothed_count = max(0, round(smoothed))

        old_length = state["queue_length"]
        state["queue_length"] = smoothed_count

        # Track for growth rate
        now = time.time()
        self._queue_lengths[queue_id].append((now, smoothed_count))

        # Detect arrivals and departures
        if smoothed_count > old_length:
            state["trend"] = "growing"
            # Record arrival
            self._arrivals[queue_id].append(now)
        elif smoothed_count < old_length:
            state["trend"] = "shrinking"
            # Record departure
            self._departures[queue_id].append(now)
        else:
            state["trend"] = "stable"

        # Update status
        if smoothed_count <= 3:
            state["status"] = "low"
        elif smoothed_count <= 7:
            state["status"] = "moderate"
        else:
            state["status"] = "high"

        state["last_updated"] = datetime.now(timezone.utc)

    def get_live_status(self, queue_id: int, db: Session) -> dict:
        """
        Get live status with ML-predicted wait time.

        The prediction uses the online ensemble that continuously learns
        from real service times observed in the queue.
        """
        state = self.get_or_create_state(queue_id)
        queue = db.query(Queue).filter(Queue.id == queue_id).first()
        tracker = self.service_trackers[queue_id]

        # Get features for ML prediction
        now = datetime.now()
        time_of_day = now.hour + now.minute / 60.0
        day_of_week = now.weekday()

        arrival_rate = self._compute_rate(self._arrivals.get(queue_id, deque()))
        departure_rate = self._compute_rate(self._departures.get(queue_id, deque()))
        avg_service_time = tracker.get_avg_service_time()
        service_variability = tracker.get_service_time_variability()
        queue_growth = self._compute_queue_growth(queue_id)

        # Online ML ensemble prediction
        prediction = self.predictor.predict(
            queue_length=state["queue_length"],
            arrival_rate=arrival_rate,
            departure_rate=departure_rate,
            mean_service_time=avg_service_time,
            active_counters=state["active_counters"],
            queue_growth_rate=queue_growth,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            service_time_variability=service_variability,
        )

        state["estimated_wait_seconds"] = prediction["predicted_wait"]
        state["lower_bound"] = prediction["lower_bound"]
        state["upper_bound"] = prediction["upper_bound"]
        state["avg_service_time"] = avg_service_time
        state["model_agreement"] = prediction.get("model_agreement", 0.5)

        return {
            "queue_id": queue_id,
            "queue_name": queue.name if queue else f"Queue {queue_id}",
            "queue_length": state["queue_length"],
            "raw_queue_length": state["raw_queue_length"],
            "estimated_wait_minutes": round(state["estimated_wait_seconds"] / 60, 1),
            "lower_bound_minutes": round(state["lower_bound"] / 60, 1) if state["lower_bound"] else None,
            "upper_bound_minutes": round(state["upper_bound"] / 60, 1) if state["upper_bound"] else None,
            "avg_service_time": round(avg_service_time, 1),
            "trend": state["trend"],
            "status": state["status"],
            "last_updated": state["last_updated"].isoformat(),
            "active_counters": state["active_counters"],
            "arrival_rate": round(arrival_rate, 2),
            "departure_rate": round(departure_rate, 2),
            "service_time_variability": round(service_variability, 1),
            "model_agreement": prediction.get("model_agreement", 0.5),
        }

    def _compute_rate(self, event_queue: deque, window: float = 60) -> float:
        """Events per minute in the last window."""
        now = time.time()
        count = sum(1 for t in event_queue if now - t < window)
        return count / (window / 60.0)

    def _compute_queue_growth(self, queue_id: int) -> float:
        """Queue length change rate per minute."""
        now = time.time()
        cutoff = now - 120
        recent = [(ts, length) for ts, length in self._queue_lengths.get(queue_id, deque()) if ts >= cutoff]
        if len(recent) < 2:
            return 0.0
        old_length = recent[0][1]
        new_length = recent[-1][1]
        dt = (recent[-1][0] - recent[0][0]) / 60.0
        if dt <= 0:
            return 0.0
        return (new_length - old_length) / dt


# ─── Database operations ────────────────────────────────────

def create_queue(db: Session, name: str, location: str = "", status: str = "active") -> Queue:
    queue = Queue(name=name, location=location, status=status)
    db.add(queue)
    db.commit()
    db.refresh(queue)
    return queue


def get_queue(db: Session, queue_id: int) -> Optional[Queue]:
    return db.query(Queue).filter(Queue.id == queue_id).first()


def list_queues(db: Session) -> List[Queue]:
    return db.query(Queue).all()


def create_counter(db: Session, queue_id: int, name: str) -> Counter:
    counter = Counter(queue_id=queue_id, name=name)
    db.add(counter)
    db.commit()
    db.refresh(counter)
    return counter


def record_observation(db: Session, data: dict) -> QueueObservation:
    obs = QueueObservation(**data)
    db.add(obs)
    db.commit()
    db.refresh(obs)
    return obs


def record_prediction(db: Session, queue_id: int, predicted: float,
                      lower: float = None, upper: float = None) -> Prediction:
    pred = Prediction(
        queue_id=queue_id,
        predicted_wait=predicted,
        lower_bound=lower,
        upper_bound=upper,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def record_anomaly(db: Session, queue_id: int, type: str, severity: str,
                   evidence: str) -> Anomaly:
    anomaly = Anomaly(
        queue_id=queue_id,
        type=type,
        severity=severity,
        evidence=evidence,
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)
    return anomaly


def get_analytics(db: Session, queue_id: int) -> dict:
    observations = (
        db.query(QueueObservation)
        .filter(QueueObservation.queue_id == queue_id)
        .all()
    )
    predictions = (
        db.query(Prediction)
        .filter(Prediction.queue_id == queue_id)
        .all()
    )
    avg_queue_length = (
        sum(o.queue_length for o in observations) / len(observations)
        if observations else 0
    )
    return {
        "queue_id": queue_id,
        "total_observations": len(observations),
        "total_predictions": len(predictions),
        "average_queue_length": round(avg_queue_length, 1),
    }


# ─── Singleton ──────────────────────────────────────────────

state_manager = QueueStateManager()
