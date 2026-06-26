from dataclasses import dataclass
from collections import deque
import math


@dataclass
class CrossingCandidate:
    frame_idx: int
    track_id: int
    bbox: object
    direction: str
    label: str
    category_confidence: float
    yolo_confidence: float
    track_age_frames: int | None = None
    event_debug: dict | None = None


class CrossingEventDeduplicator:
    """
    Suppresses duplicate crossing events caused by duplicated boxes/tracks
    on the same physical vehicle.

    This does not merge normal neighboring vehicles unless their boxes are
    almost identical and cross in the same direction within a short time window.
    """

    def __init__(
        self,
        iou_threshold: float = 0.85,
        center_distance_threshold: float = 30.0,
        frame_window: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.center_distance_threshold = center_distance_threshold
        self.frame_window = frame_window
        self.recent_accepted = deque()

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter
        if union <= 0:
            return 0.0

        return inter / union

    @staticmethod
    def _center_distance(a, b) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)

        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        return math.hypot(acx - bcx, acy - bcy)

    def _is_duplicate(self, candidate: CrossingCandidate, accepted: CrossingCandidate) -> tuple[bool, dict]:
        if candidate.direction != accepted.direction:
            return False, {}

        frame_gap = abs(candidate.frame_idx - accepted.frame_idx)
        if frame_gap > self.frame_window:
            return False, {}

        iou = self._bbox_iou(candidate.bbox, accepted.bbox)
        center_distance = self._center_distance(candidate.bbox, accepted.bbox)

        duplicate = (
            iou >= self.iou_threshold
            or (
                iou >= 0.50
                and center_distance <= self.center_distance_threshold
            )
        )

        details = {
            "duplicate_iou": iou,
            "duplicate_center_distance": center_distance,
            "duplicate_frame_gap": frame_gap,
            "duplicate_of_track_id": accepted.track_id,
        }

        return duplicate, details

    @staticmethod
    def _candidate_priority(candidate: CrossingCandidate) -> tuple:
        """
        Higher is better.

        Prefer:
        1. older track,
        2. higher detector confidence,
        3. higher classification confidence.
        """
        age = candidate.track_age_frames if candidate.track_age_frames is not None else 0
        return (
            age,
            candidate.yolo_confidence,
            candidate.category_confidence,
        )

    def filter_frame_candidates(self, candidates: list[CrossingCandidate]) -> tuple[list[CrossingCandidate], list[CrossingCandidate]]:
        """
        Filter crossing candidates from one frame.

        Returns:
        - accepted candidates,
        - suppressed duplicate candidates.
        """

        # Remove stale accepted events from history.
        if candidates:
            current_frame = max(c.frame_idx for c in candidates)
            while self.recent_accepted and current_frame - self.recent_accepted[0].frame_idx > self.frame_window:
                self.recent_accepted.popleft()

        # Prefer stable/older tracks first.
        ordered = sorted(candidates, key=self._candidate_priority, reverse=True)

        accepted_now = []
        suppressed = []

        for candidate in ordered:
            is_dup = False
            duplicate_details = {}

            # Compare with same-frame accepted candidates and recent accepted history.
            for accepted in list(accepted_now) + list(self.recent_accepted):
                is_dup, duplicate_details = self._is_duplicate(candidate, accepted)
                if is_dup:
                    break

            if is_dup:
                if candidate.event_debug is not None:
                    candidate.event_debug["event_suppressed"] = True
                    candidate.event_debug["suppressed_reason"] = "duplicate_overlapping_crossing_event"
                    candidate.event_debug.update(duplicate_details)
                suppressed.append(candidate)
            else:
                if candidate.event_debug is not None:
                    candidate.event_debug["event_suppressed"] = False
                    candidate.event_debug["suppressed_reason"] = ""
                accepted_now.append(candidate)
                self.recent_accepted.append(candidate)

        # Preserve readable order by track_id/frame if desired.
        accepted_now = sorted(accepted_now, key=lambda c: c.track_id)
        suppressed = sorted(suppressed, key=lambda c: c.track_id)

        return accepted_now, suppressed