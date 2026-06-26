from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackObservation:
    frame_idx: int
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    crossed: int = 0
    direction: str = ""


@dataclass
class CrossingEvent:
    frame_idx: int
    track_id: int
    gddkia_class: str
    direction: str