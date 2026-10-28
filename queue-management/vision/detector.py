"""
Person detector using YOLOv8.

Balanced for laptop webcams: detects people at normal distances
while filtering out obvious false positives (tiny objects, screen reflections).
"""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        min_bbox_height: int = 40,
        nms_iou_threshold: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_bbox_height = min_bbox_height
        self.nms_iou_threshold = nms_iou_threshold
        self.model = YOLO(model_path)
        self.person_class_id = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect people in a single frame."""
        h, frame_w = frame.shape[:2]
        min_h = int(self.min_bbox_height * (h / 480))

        results = self.model(frame, verbose=False)
        raw_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != self.person_class_id:
                    continue
                if conf < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                bbox_h = y2 - y1
                if bbox_h < min_h:
                    continue

                raw_detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="person",
                    )
                )

        return self._nms(raw_detections)

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        for det in sorted_dets:
            is_duplicate = False
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou >= self.nms_iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(det)

        return keep

    @staticmethod
    def _compute_iou(b1, b2) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _point_in_polygon(point, polygon) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
