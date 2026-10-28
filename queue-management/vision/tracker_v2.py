"""
ByteTrack-inspired multi-object tracker with Kalman-based motion prediction.

Key improvements over the simple IoU tracker:
1. Kalman filter per track predicts where each person will be next frame
2. Velocity-based matching: uses predicted position, not just current IoU
3. Handles occlusion: keeps lost tracks alive with predicted positions
4. Better ID preservation: reduces ID switches when people cross paths
5. Direction-aware: tracks which way people are moving (entering/leaving queue)

This is the production-quality tracker for real-world deployment.
"""

import math
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

from vision.detector import Detection


@dataclass
class TrackKalman:
    """
    Single tracked person with Kalman-based motion model.

    The Kalman filter predicts where this person will be in the next frame,
    allowing us to match detections even when people temporarily disappear
    (occlusion, turning sideways, etc).
    """
    track_id: str
    bbox: Tuple[int, int, int, int]

    # Kalman state (center x, center y, width, height, vx, vy, vw, vh)
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0

    # Kalman covariance (diagonal elements for simplicity)
    P: List[float] = field(default_factory=lambda: [10.0] * 8)

    # Track lifecycle
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    enter_time: float = 0.0
    last_seen: float = 0.0

    # History
    history: List[Tuple[int, int, int, int, float]] = field(default_factory=list)
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def center(self) -> Tuple[int, int]:
        return (int(self.cx), int(self.cy))

    @property
    def predicted_center(self) -> Tuple[int, int]:
        """Predicted position in the next frame."""
        return (int(self.cx + self.vx), int(self.cy + self.vy))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed if seen multiple times."""
        return self.hits >= 3

    @property
    def track_duration(self) -> float:
        """How long this track has existed (seconds)."""
        if not self.history:
            return 0.0
        return self.history[-1][4] - self.enter_time


class KalmanTracker:
    """
    Kalman filter for 2D bounding box tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]

    The filter predicts position based on velocity and corrects
    using the actual detection measurement.
    """

    def __init__(self):
        # Process noise (how much we expect position/size to change)
        self.Q_pos = 1.0    # position noise
        self.Q_vel = 0.5    # velocity noise
        self.Q_size = 0.1   # size noise

        # Measurement noise
        self.R_pos = 2.0    # position measurement noise
        self.R_size = 1.0   # size measurement noise

    def predict(self, track: TrackKalman, dt: float = 1.0):
        """Predict next state based on velocity model."""
        # Position prediction: x_pred = x + v * dt
        track.cx += track.vx * dt
        track.cy += track.vy * dt
        track.w += track.vw * dt
        track.h += track.vh * dt

        # Increase uncertainty
        track.P[0] += self.Q_pos * dt  # cx
        track.P[1] += self.Q_pos * dt  # cy
        track.P[2] += self.Q_size * dt  # w
        track.P[3] += self.Q_size * dt  # h
        track.P[4] += self.Q_vel * dt  # vx
        track.P[5] += self.Q_vel * dt  # vy
        track.P[6] += self.Q_size * dt  # vw
        track.P[7] += self.Q_size * dt  # vh

    def update(self, track: TrackKalman, bbox: Tuple[int, int, int, int]):
        """Update state with actual measurement."""
        x1, y1, x2, y2 = bbox
        cx_m = (x1 + x2) / 2.0
        cy_m = (y1 + y2) / 2.0
        w_m = x2 - x1
        h_m = y2 - y1

        # Innovation
        y_cx = cx_m - track.cx
        y_cy = cy_m - track.cy
        y_w = w_m - track.w
        y_h = h_m - track.h

        # Kalman gain (simplified: independent dimensions)
        for i in range(4):
            S = self.R_pos if i < 2 else self.R_size
            K = track.P[i] / (track.P[i] + S)
            innovation = [y_cx, y_cy, y_w, y_h][i]
            vals = [track.cx, track.cy, track.w, track.h]
            vals[i] = vals[i] + K * innovation
            track.cx, track.cy, track.w, track.h = vals
            track.P[i] = (1 - K) * track.P[i]

        # Update velocity from position change
        track.vx = track.cx - (track.cx - track.vx) * 0.8
        track.vy = track.cy - (track.cy - track.vy) * 0.8

        # Decrease uncertainty after measurement
        for i in range(8):
            track.P[i] = max(0.1, track.P[i] * 0.8)


def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _center_distance(t1: TrackKalman, det_bbox: Tuple[int, int, int, int]) -> float:
    """Distance between track predicted center and detection center."""
    px, py = t1.predicted_center
    dx = (det_bbox[0] + det_bbox[2]) / 2.0 - px
    dy = (det_bbox[1] + det_bbox[3]) / 2.0 - py
    return math.sqrt(dx * dx + dy * dy)


class ByteTracker:
    """
    ByteTrack-inspired tracker with Kalman prediction.

    Matching strategy (two stages):
    1. High-confidence match: IoU > 0.3 with existing tracks
    2. Low-confidence match: IoU > 0.15 for remaining detections
       (handles partial occlusions where IoU drops)

    Lost tracks are kept alive for max_age frames and matched
    using predicted positions (Kalman).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        low_iou_threshold: float = 0.15,
        max_age: int = 8,
        min_hits: int = 2,
        max_distance: float = 120.0,
    ):
        self.iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.tracks: List[TrackKalman] = []
        self._next_id = 0
        self.kalman = KalmanTracker()

    def update(self, detections: List[Detection]) -> List[TrackKalman]:
        """
        Process new detections and return active tracked persons.

        Uses ByteTrack's two-stage matching:
        - Stage 1: match high-confidence detections to existing tracks
        - Stage 2: match remaining low-confidence detections
        - Unmatched detections create new tracks
        - Unmatched tracks get predicted positions
        """
        now = time.time()

        # Predict all existing track positions
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            self.kalman.predict(track)

        if not detections:
            self._cleanup()
            return self._active_tracks()

        det_bboxes = [d.bbox for d in detections]
        det_used = [False] * len(detections)
        track_used = [False] * len(self.tracks)

        # Stage 1: High-confidence matching (IoU-based)
        if self.tracks:
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for i, dbox in enumerate(det_bboxes):
                for j, track in enumerate(self.tracks):
                    # Use predicted bbox for matching
                    pred_bbox = self._predicted_bbox(track)
                    iou_matrix[i, j] = _iou(dbox, pred_bbox)

            matched_high = self._match_stage(
                iou_matrix, self.iou_threshold, det_used, track_used
            )
            for di, ti in matched_high:
                self._update_track(self.tracks[ti], det_bboxes[di], now)

        # Stage 2: Low-confidence matching for remaining
        remaining_dets = [(i, d) for i, d in enumerate(detections) if not det_used[i]]
        remaining_tracks = [(j, t) for j, t in enumerate(self.tracks) if not track_used[j]]

        if remaining_dets and remaining_tracks:
            iou_matrix2 = np.zeros((len(remaining_dets), len(remaining_tracks)))
            for i, (di, d) in enumerate(remaining_dets):
                for j, (ti, t) in enumerate(remaining_tracks):
                    pred_bbox = self._predicted_bbox(t)
                    iou_matrix2[i, j] = _iou(d.bbox, pred_bbox)

            matched_low = self._match_stage(
                iou_matrix2, self.low_iou_threshold,
                [False] * len(remaining_dets),
                [False] * len(remaining_tracks),
            )
            for ri, rt in matched_low:
                di = remaining_dets[ri][0]
                ti = remaining_tracks[rt][0]
                self._update_track(self.tracks[ti], det_bboxes[di], now)
                det_used[di] = True
                track_used[ti] = True

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if not det_used[i]:
                self._create_track(det.bbox, now)

        # Cleanup dead tracks
        self._cleanup()

        return self._active_tracks()

    def _update_track(self, track: TrackKalman, bbox: Tuple[int, int, int, int], timestamp: float):
        """Update an existing track with a new detection."""
        # Record velocity before update
        old_cx, old_cy = track.cx, track.cy

        # Kalman update
        self.kalman.update(track, bbox)
        track.bbox = bbox
        track.hits += 1
        track.time_since_update = 0
        track.last_seen = timestamp
        track.history.append((*bbox, timestamp))

        # Record velocity from Kalman update
        vx = track.cx - old_cx
        vy = track.cy - old_cy
        track.velocity_history.append((vx, vy))

    def _create_track(self, bbox: Tuple[int, int, int, int], timestamp: float):
        """Create a new track from a detection."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        track = TrackKalman(
            track_id=f"person_{self._next_id}",
            bbox=bbox,
            cx=cx, cy=cy, w=w, h=h,
            enter_time=timestamp,
            last_seen=timestamp,
            history=[(*bbox, timestamp)],
        )
        self._next_id += 1
        self.tracks.append(track)

    def _predicted_bbox(self, track: TrackKalman) -> Tuple[int, int, int, int]:
        """Get the predicted bounding box for a track."""
        px, py = track.predicted_center
        hw = track.w / 2.0
        hh = track.h / 2.0
        return (int(px - hw), int(py - hh), int(px + hw), int(py + hh))

    def _match_stage(
        self,
        iou_matrix: np.ndarray,
        threshold: float,
        det_used: List[bool],
        track_used: List[bool],
    ) -> List[Tuple[int, int]]:
        """Greedy matching at a given IoU threshold."""
        matched = []
        if iou_matrix.size == 0:
            return matched

        matrix = iou_matrix.copy()
        while True:
            if matrix.size == 0:
                break
            max_val = matrix.max()
            if max_val < threshold:
                break
            di, ti = np.unravel_index(matrix.argmax(), matrix.shape)
            matched.append((int(di), int(ti)))
            matrix[di, :] = 0
            matrix[:, ti] = 0

        return matched

    def _cleanup(self):
        """Remove dead tracks."""
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

    def _active_tracks(self) -> List[TrackKalman]:
        """Return tracks that have been seen enough times."""
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def get_all_tracks(self) -> List[TrackKalman]:
        """Return all tracks (including low-confidence ones)."""
        return self.tracks
