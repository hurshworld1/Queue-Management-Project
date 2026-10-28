import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from vision.detector import PersonDetector, Detection
from vision.tracker import PersonTracker, Track
from vision.roi import ROICreator, QueueZone
from vision.event_extractor import EventExtractor


class TestPersonDetector:
    def test_detection_dataclass(self):
        det = Detection(bbox=(10, 20, 100, 200), confidence=0.85, class_id=0, class_name="person")
        assert det.bbox == (10, 20, 100, 200)
        assert det.confidence == 0.85

    def test_point_in_polygon(self):
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert PersonDetector._point_in_polygon((50, 50), polygon) is True
        assert PersonDetector._point_in_polygon((150, 50), polygon) is False

    def test_point_in_triangle(self):
        polygon = [(0, 0), (100, 0), (50, 100)]
        assert PersonDetector._point_in_polygon((50, 30), polygon) is True
        assert PersonDetector._point_in_polygon((0, 80), polygon) is False


class TestPersonTracker:
    def test_create_tracks_from_detections(self):
        tracker = PersonTracker(min_hits=1)
        detections = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=(60, 60, 100, 100), confidence=0.8, class_id=0, class_name="person"),
        ]
        tracks = tracker.update(detections)
        assert len(tracker.tracks) == 2

    def test_iou_computation(self):
        bboxes_a = [(0, 0, 100, 100)]
        bboxes_b = [(50, 50, 150, 150)]
        iou = PersonTracker._compute_iou_matrix(bboxes_a, bboxes_b)
        # Intersection area = 50*50 = 2500, Union = 100*100 + 100*100 - 2500 = 17500
        expected_iou = 2500 / 17500
        assert abs(iou[0, 0] - expected_iou) < 0.001

    def test_track_center(self):
        track = Track(track_id="t1", bbox=(0, 0, 100, 200))
        assert track.center == (50, 100)


class TestROICreator:
    def test_add_zone(self):
        roi = ROICreator()
        zone = roi.add_queue_zone("queue1", [(0, 0), (100, 0), (100, 100), (0, 100)])
        assert zone.name == "queue1"
        assert len(roi.zones) == 1

    def test_point_in_zone(self):
        roi = ROICreator()
        roi.add_queue_zone("queue1", [(0, 0), (100, 0), (100, 100), (0, 100)])
        assert roi.point_in_zone((50, 50), "queue1") is True
        assert roi.point_in_zone((150, 50), "queue1") is False

    def test_which_zone(self):
        roi = ROICreator()
        roi.add_queue_zone("queue1", [(0, 0), (100, 0), (100, 100), (0, 100)])
        roi.add_queue_zone("queue2", [(200, 0), (300, 0), (300, 100), (200, 100)])
        zone = roi.which_zone((50, 50))
        assert zone is not None and zone.name == "queue1"
        zone2 = roi.which_zone((250, 50))
        assert zone2 is not None and zone2.name == "queue2"

    def test_to_dict_and_from_dict(self):
        roi = ROICreator()
        roi.add_queue_zone("q1", [(0, 0), (10, 0), (10, 10)])
        data = roi.to_dict()
        roi2 = ROICreator()
        roi2.from_dict(data)
        assert len(roi2.zones) == 1
        assert roi2.zones[0].name == "q1"


class TestEventExtractor:
    def test_no_events_empty_tracks(self):
        roi = ROICreator()
        roi.add_queue_zone("queue1", [(0, 0), (100, 0), (100, 100), (0, 100)])
        extractor = EventExtractor(roi, queue_id=1)
        events = extractor.process_tracks([])
        assert len(events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
