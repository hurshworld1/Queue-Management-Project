from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from vision.tracker import Track
from vision.roi import ROICreator, QueueZone


@dataclass
class QueueEvent:
    event_type: str  # arrival, departure, service_start, service_end
    person_temp_id: str
    timestamp: float
    queue_id: Optional[int] = None
    counter_id: Optional[int] = None
    zone_name: str = ""


class EventExtractor:
    """Extract queue entry, departure, and service events from tracked trajectories."""

    def __init__(self, roi_creator: ROICreator, queue_id: int = 1):
        self.roi = roi_creator
        self.queue_id = queue_id
        self._person_states: Dict[str, str] = {}  # track_id -> state
        self._person_zones: Dict[str, str] = {}  # track_id -> current zone
        self.events: List[QueueEvent] = []

    def process_tracks(self, tracks: List[Track]) -> List[QueueEvent]:
        """Process tracked persons and extract queue events."""
        new_events = []
        active_ids = {t.track_id for t in tracks}

        for track in tracks:
            tid = track.track_id
            current_zone = self.roi.which_zone(track.center)
            zone_name = current_zone.name if current_zone else "none"
            prev_zone = self._person_zones.get(tid, "none")

            # Detect zone transitions
            if prev_zone != zone_name:
                event = self._detect_event(tid, prev_zone, zone_name, track)
                if event:
                    new_events.append(event)
                    self.events.append(event)

            self._person_zones[tid] = zone_name

        # Clean up departed tracks
        departed = set(self._person_states.keys()) - active_ids
        for tid in departed:
            if self._person_states.get(tid) in ("queue", "service"):
                event = QueueEvent(
                    event_type="departure",
                    person_temp_id=tid,
                    timestamp=time.time(),
                    queue_id=self.queue_id,
                )
                new_events.append(event)
                self.events.append(event)
            del self._person_states[tid]
            self._person_zones.pop(tid, None)

        return new_events

    def _detect_event(
        self, track_id: str, from_zone: str, to_zone: str, track: Track
    ) -> Optional[QueueEvent]:
        """Detect an event based on zone transition."""
        now = track.history[-1][4] if track.history else time.time()

        # Entering queue zone
        if to_zone.startswith("queue") and from_zone == "none":
            self._person_states[track_id] = "queue"
            return QueueEvent(
                event_type="arrival",
                person_temp_id=track_id,
                timestamp=now,
                queue_id=self.queue_id,
                zone_name=to_zone,
            )

        # Moving to service zone
        if to_zone.startswith("service") and from_zone.startswith("queue"):
            self._person_states[track_id] = "service"
            counter_id = self._get_counter_from_zone(to_zone)
            return QueueEvent(
                event_type="service_start",
                person_temp_id=track_id,
                timestamp=now,
                queue_id=self.queue_id,
                counter_id=counter_id,
                zone_name=to_zone,
            )

        # Leaving service zone (service complete)
        if from_zone.startswith("service") and not to_zone.startswith("service"):
            self._person_states[track_id] = "done"
            counter_id = self._get_counter_from_zone(from_zone)
            return QueueEvent(
                event_type="service_end",
                person_temp_id=track_id,
                timestamp=now,
                queue_id=self.queue_id,
                counter_id=counter_id,
                zone_name=from_zone,
            )

        return None

    def _get_counter_from_zone(self, zone_name: str) -> Optional[int]:
        zone = self.roi.get_zone(zone_name)
        return zone.counter_id if zone else None

    def get_queue_events(self) -> List[QueueEvent]:
        return [e for e in self.events if e.event_type in ("arrival", "departure")]

    def get_service_events(self) -> List[QueueEvent]:
        return [e for e in self.events if e.event_type in ("service_start", "service_end")]

    def get_recent_events(self, seconds: float = 60) -> List[QueueEvent]:
        cutoff = time.time() - seconds
        return [e for e in self.events if e.timestamp >= cutoff]
