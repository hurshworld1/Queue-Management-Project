from typing import List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class QueueZone:
    """A defined region for a queue or service area."""
    name: str
    polygon: List[Tuple[int, int]]
    zone_type: str = "queue"  # queue, service, entrance, exit
    counter_id: Optional[int] = None


class ROICreator:
    """Create and manage Regions of Interest for queue monitoring."""

    def __init__(self):
        self.zones: List[QueueZone] = []

    def add_queue_zone(
        self,
        name: str,
        polygon: List[Tuple[int, int]],
        counter_id: Optional[int] = None,
    ) -> QueueZone:
        zone = QueueZone(name=name, polygon=polygon, zone_type="queue", counter_id=counter_id)
        self.zones.append(zone)
        return zone

    def add_entrance_zone(
        self, name: str, polygon: List[Tuple[int, int]]
    ) -> QueueZone:
        zone = QueueZone(name=name, polygon=polygon, zone_type="entrance")
        self.zones.append(zone)
        return zone

    def add_exit_zone(
        self, name: str, polygon: List[Tuple[int, int]]
    ) -> QueueZone:
        zone = QueueZone(name=name, polygon=polygon, zone_type="exit")
        self.zones.append(zone)
        return zone

    def add_service_zone(
        self, name: str, polygon: List[Tuple[int, int]], counter_id: int
    ) -> QueueZone:
        zone = QueueZone(name=name, polygon=polygon, zone_type="service", counter_id=counter_id)
        self.zones.append(zone)
        return zone

    def get_zone(self, name: str) -> Optional[QueueZone]:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def point_in_zone(self, point: Tuple[int, int], zone_name: str) -> bool:
        zone = self.get_zone(zone_name)
        if zone is None:
            return False
        return self._point_in_polygon(point, zone.polygon)

    def which_zone(self, point: Tuple[int, int]) -> Optional[QueueZone]:
        for zone in self.zones:
            if self._point_in_polygon(point, zone.polygon):
                return zone
        return None

    @staticmethod
    def _point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
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

    def to_dict(self) -> List[dict]:
        return [
            {
                "name": z.name,
                "polygon": z.polygon,
                "zone_type": z.zone_type,
                "counter_id": z.counter_id,
            }
            for z in self.zones
        ]

    def from_dict(self, data: List[dict]):
        self.zones = [
            QueueZone(
                name=z["name"],
                polygon=[tuple(p) for p in z["polygon"]],
                zone_type=z.get("zone_type", "queue"),
                counter_id=z.get("counter_id"),
            )
            for z in data
        ]

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.from_dict(data)
