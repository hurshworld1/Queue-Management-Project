"""
1D Kalman Filter for smoothing noisy real-time measurements.

Used to smooth:
- Queue counts from camera detection (noisy, frame-by-frame)
- Estimated wait times (jittery from raw calculations)
- Service time measurements (variability between individuals)

The filter separates signal from noise using a state-space model:
  state(t) = state(t-1) + velocity(t-1) * dt + noise
  measurement(t) = state(t) + noise

This gives smooth, responsive estimates that don't lag behind real changes.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanState:
    """Internal state of a 1D Kalman filter."""
    x: float = 0.0          # estimated value
    v: float = 0.0          # estimated velocity (rate of change)
    P_x: float = 100.0      # estimate uncertainty
    P_v: float = 100.0      # velocity uncertainty
    P_xv: float = 0.0       # cross-covariance
    initialized: bool = False
    last_time: float = 0.0


class KalmanFilter1D:
    """
    1D Kalman filter with constant-velocity model.

    Parameters:
        process_noise: How much we expect the value to change between updates.
            Higher = more responsive but noisier. Lower = smoother but laggier.
        measurement_noise: How noisy we think each measurement is.
            Higher = trust measurements less (smoother). Lower = trust more (noisier).
        dt: Expected time between updates (seconds). Used to scale process noise.
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 2.0,
        dt: float = 0.2,
    ):
        self.Q = process_noise       # process noise variance
        self.R = measurement_noise   # measurement noise variance
        self.dt = dt
        self.state = KalmanState()

    def update(self, measurement: float, timestamp: float = None) -> float:
        """
        Feed a new measurement, return the filtered estimate.

        Args:
            measurement: Raw measured value (e.g., queue count from camera).
            timestamp: Current time in seconds. If None, uses time.time().

        Returns:
            Smoothed estimate.
        """
        now = timestamp or time.time()
        s = self.state

        if not s.initialized:
            s.x = measurement
            s.v = 0.0
            s.P_x = 10.0
            s.P_v = 10.0
            s.P_xv = 0.0
            s.last_time = now
            s.initialized = True
            return s.x

        # Time step
        dt = max(now - s.last_time, 0.01)
        s.last_time = now

        # --- Predict ---
        # state prediction: x_pred = x + v * dt
        x_pred = s.x + s.v * dt
        v_pred = s.v

        # Uncertainty prediction
        P_x_pred = s.P_x + dt * (s.P_v * dt + 2 * s.P_xv) + self.Q * dt * dt
        P_v_pred = s.P_v + self.Q
        P_xv_pred = s.P_xv + s.P_v * dt

        # --- Update ---
        # Innovation (measurement residual)
        y = measurement - x_pred

        # Innovation covariance
        S = P_x_pred + self.R

        # Kalman gain
        K_x = P_x_pred / S
        K_v = P_xv_pred / S

        # State update
        s.x = x_pred + K_x * y
        s.v = v_pred + K_v * y

        # Covariance update
        s.P_x = (1 - K_x) * P_x_pred
        s.P_v = (1 - K_v) * P_v_pred
        s.P_xv = (1 - K_x) * P_xv_pred

        return s.x

    @property
    def estimate(self) -> float:
        """Current filtered estimate."""
        return self.state.x

    @property
    def velocity(self) -> float:
        """Estimated rate of change per second."""
        return self.state.v

    @property
    def uncertainty(self) -> float:
        """Current estimate uncertainty (standard deviation)."""
        return math.sqrt(max(0, self.state.P_x))

    def reset(self):
        """Reset the filter to uninitialized state."""
        self.state = KalmanState()

    def predict_ahead(self, seconds: float) -> float:
        """Predict what the value will be N seconds from now."""
        return self.state.x + self.state.v * seconds


class AdaptiveKalmanFilter(KalmanFilter1D):
    """
    Kalman filter that automatically tunes its noise parameters based on
    the innovation sequence (difference between predicted and actual measurements).

    This makes it responsive to sudden changes (person walks in) while
    staying smooth during steady state (queue not changing).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._innovations = deque(maxlen=50)
        self._adaptation_rate = 0.01

    def update(self, measurement: float, timestamp: float = None) -> float:
        now = timestamp or time.time()
        s = self.state

        if not s.initialized:
            return super().update(measurement, timestamp)

        # Compute innovation before update
        dt = max(now - s.last_time, 0.01)
        x_pred = s.x + s.v * dt
        innovation = measurement - x_pred
        self._innovations.append(innovation)

        # Adapt measurement noise based on innovation variance
        if len(self._innovations) >= 10:
            innov_var = sum(i * i for i in self._innovations) / len(self._innovations)
            # If innovations are large, measurements are noisier than expected
            self.R = max(0.5, self.R * (1 - self._adaptation_rate) +
                        innov_var * self._adaptation_rate)

        return super().update(measurement, timestamp)


class ExponentialMovingAverage:
    """
    EMA for smoothing values with a specified half-life.

    Simpler than Kalman but good for cases where we just want
    exponential smoothing without velocity estimation.
    """

    def __init__(self, half_life: float = 5.0):
        """
        Args:
            half_life: Time in seconds for old data to lose half its weight.
        """
        self.half_life = half_life
        self._alpha = 1.0 - math.exp(-math.log(2) / half_life)
        self._value: Optional[float] = None
        self._last_time: Optional[float] = None

    def update(self, value: float, timestamp: float = None) -> float:
        now = timestamp or time.time()
        if self._value is None:
            self._value = value
            self._last_time = now
            return self._value

        # Adjust alpha based on actual time elapsed
        dt = max(now - self._last_time, 0.01)
        alpha = 1.0 - math.exp(-math.log(2) * dt / self.half_life)

        self._value = alpha * value + (1 - alpha) * self._value
        self._last_time = now
        return self._value

    @property
    def value(self) -> float:
        return self._value or 0.0

    def reset(self):
        self._value = None
        self._last_time = None
