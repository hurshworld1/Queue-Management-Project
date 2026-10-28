import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
import uuid

from vision.detector import Detection


@dataclass
class Track:
    track_id: str
    bbox: Tuple[int, int, int, int]
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[int, int, int, int, float]] = field(default_factory=list)
    enter_time: float = 0.0
    last_seen: float = 0.0

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class PersonTracker:
    """IoU-based multi-object tracker inspired by ByteTrack."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 0
        self._lost_tracks: List[Track] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections, return active tracks."""
        now = time.time()

        # Age all existing tracks
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        if not detections:
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits]

        # Compute IoU cost matrix
        det_bboxes = [d.bbox for d in detections]
        track_bboxes = [t.bbox for t in self.tracks]

        if not track_bboxes:
            # No existing tracks, create new ones
            for det in detections:
                self._create_track(det, now)
            return []

        iou_matrix = self._compute_iou_matrix(det_bboxes, track_bboxes)

        # Greedy matching
        matched_dets, matched_tracks = self._greedy_match(iou_matrix)

        # Update matched tracks
        for det_idx, track_idx in matched_dets:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.bbox = det.bbox
            track.hits += 1
            track.time_since_update = 0
            track.history.append((*det.bbox, now))
            track.last_seen = now

        # Create new tracks for unmatched detections
        matched_det_indices = {d for d, _ in matched_dets}
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self._create_track(det, now)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _create_track(self, detection: Detection, timestamp: float):
        track = Track(
            track_id=f"person_{self._next_id}",
            bbox=detection.bbox,
            enter_time=timestamp,
            last_seen=timestamp,
            history=[(*detection.bbox, timestamp)],
        )
        self._next_id += 1
        self.tracks.append(track)

    @staticmethod
    def _compute_iou_matrix(
        bboxes_a: List[Tuple[int, int, int, int]],
        bboxes_b: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of bounding boxes."""
        matrix = np.zeros((len(bboxes_a), len(bboxes_b)))

        for i, (ax1, ay1, ax2, ay2) in enumerate(bboxes_a):
            for j, (bx1, by1, bx2, by2) in enumerate(bboxes_b):
                # Intersection
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)

                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter

                matrix[i, j] = inter / union if union > 0 else 0.0

        return matrix

    def _greedy_match(
        self, iou_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Greedy matching based on IoU."""
        matched_dets = []
        matched_tracks = set()

        if iou_matrix.size == 0:
            return [], set()

        while True:
            if iou_matrix.size == 0:
                break

            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_dets.append((int(det_idx), int(track_idx)))
            matched_tracks.add(int(track_idx))

            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0

        return matched_dets, matched_tracks

    def get_track_by_id(self, track_id: str) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
