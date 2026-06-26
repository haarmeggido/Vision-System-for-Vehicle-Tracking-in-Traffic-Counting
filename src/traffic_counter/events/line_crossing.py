import math
import numpy as np


def line_side(px, py, ax, ay, bx, by):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


class LineCrossCounter:
    """
    Single-line crossing event engine.

    Current baseline logic:
    - uses bbox centroid
    - detects sign change across the line
    - counts a track only once

    Includes debug logic for exact line crossing detection and distance calculation.
    Includes optional logic for same id recounting
    """

    def __init__(self, p1, p2, reference_point: str = "centroid", allow_same_id_recount: bool = False, same_id_recount_cooldown: int = 30):
        self.p1 = p1
        self.p2 = p2
        self.reference_point = reference_point
        self.allow_same_id_recount = allow_same_id_recount
        self.same_id_recount_cooldown = same_id_recount_cooldown

        self.last_side = {}
        self.last_nonzero_side = {}
        self.counted = set()

        self.last_counted_frame = {}
        self.last_counted_direction = {}
        self.same_id_recount_events = 0

        self.stats = {"A->B": 0, "B->A": 0}

    def get_reference_point(self, bbox):
        x1, y1, x2, y2 = map(float, bbox)
        if self.reference_point == "centroid":
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        elif self.reference_point == "bottom_center":
            return (x1 + x2) / 2.0, y2
        
        raise ValueError(f"Unsupported reference point: {self.reference_point}")

    def _line_length(self):
        ax, ay = self.p1
        bx, by = self.p2
        return math.hypot(bx - ax, by - ay)
    
    def update(self, track_id: int, bbox, frame_idx: int | None = None):
        ref_x, ref_y = self.get_reference_point(bbox)
        raw_side = line_side(ref_x, ref_y, *self.p1, *self.p2)
        side_now = int(np.sign(raw_side))

        prev = self.last_side.get(track_id, side_now)
        prev_nonzero = self.last_nonzero_side.get(
            track_id,
            side_now if side_now != 0 else 0,
        ) 

        counted_before = track_id in self.counted

        crossed = 0
        direction = ""
        suppressed_reason = ""
        same_id_recount = False

        would_cross_raw = (
            prev != 0
            and side_now != 0
            and int(np.sign(prev)) != int(np.sign(side_now))
        )

        would_cross_nonzero = (
            prev_nonzero != 0
            and side_now != 0
            and int(np.sign(prev_nonzero)) != int(np.sign(side_now))
        )

        actual_cross_candidate = would_cross_raw
        direction_source_side = prev
        direction_candidate = ""

        if actual_cross_candidate:
            direction = "A->B" if direction_source_side < 0 and side_now > 0 else "B->A"

            if not counted_before:
                self.counted.add(track_id)
                self.stats[direction] += 1
                self.last_counted_frame[track_id] = frame_idx
                self.last_counted_direction[track_id] = direction
                crossed = 1

            else:
                last_frame = self.last_counted_frame.get(track_id)
                last_direction = self.last_counted_direction.get(track_id)

                if frame_idx is not None and last_frame is not None:
                    frames_since_counted = frame_idx - last_frame
                else:
                    frames_since_counted = None

                can_recount = (
                    self.allow_same_id_recount
                    and frames_since_counted is not None
                    and frames_since_counted >= self.same_id_recount_cooldown
                    and last_direction is not None
                    and direction != last_direction
                )

                if can_recount:
                    self.stats[direction] += 1
                    self.last_counted_frame[track_id] = frame_idx
                    self.last_counted_direction[track_id] = direction
                    self.same_id_recount_events += 1
                    crossed = 1
                    same_id_recount = True
                else:
                    suppressed_reason = "already_counted_track_id"

        self.last_side[track_id] = side_now

        if side_now != 0:
            self.last_nonzero_side[track_id] = side_now

        line_len = self._line_length()
        line_distance_px = raw_side / line_len if line_len > 0 else 0.0

        last_counted_frame = self.last_counted_frame.get(track_id)
        last_counted_direction = self.last_counted_direction.get(track_id)

        if frame_idx is not None and last_counted_frame is not None:
            frames_since_counted = frame_idx - last_counted_frame
        else:
            frames_since_counted = None

        debug = {
            "ref_x": ref_x,
            "ref_y": ref_y,
            "line_side_raw": raw_side,
            "line_distance_px": line_distance_px,
            "side_now": side_now,
            "prev_side": int(prev),
            "prev_nonzero_side": int(prev_nonzero),
            "would_cross_using_raw_logic": bool(would_cross_raw),
            "would_cross_using_nonzero_logic": bool(would_cross_nonzero),
            "counted_before": bool(counted_before),
            "crossed": bool(crossed),
            "direction": direction,
            "counted_after": bool(track_id in self.counted),
            "reference_point": self.reference_point,

            "last_counted_frame": last_counted_frame,
            "last_counted_direction": last_counted_direction,
            "frames_since_counted": frames_since_counted,
            "suppressed_reason": suppressed_reason,
            "same_id_recount": bool(same_id_recount),
            "allow_same_id_recount": bool(self.allow_same_id_recount),
            "same_id_recount_cooldown": self.same_id_recount_cooldown,
            "actual_cross_candidate": bool(actual_cross_candidate),
            "direction_candidate": direction_candidate,
        }

        return crossed, direction, debug

    def undo_count(self, track_id: int, direction: str, frame_idx: int | None = None):
        if direction in self.stats and self.stats[direction] > 0:
            self.stats[direction] -= 1

        last_frame = self.last_counted_frame.get(track_id)
        last_direction = self.last_counted_direction.get(track_id)

        if last_direction == direction and (frame_idx is None or last_frame == frame_idx):
            self.counted.discard(track_id)
            self.last_counted_direction.pop(track_id, None)
            self.last_counted_frame.pop(track_id, None)
    
    def clear_track_state(self, track_id: int):
        self.last_side.pop(track_id, None)
        self.last_nonzero_side.pop(track_id, None)