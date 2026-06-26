import csv
import json
from pathlib import Path
from typing import Any


def _json_safe_dump(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _fmt_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _fmt_bool(value: Any) -> int:
    return int(bool(value))


class ObservationDebugWriter:
    """
    Per-frame, per-track diagnostic writer.

    Intended for debugging:
    - line-crossing state,
    - tracker ID reuse / hijacking,
    - duplicate suppression,
    - same-ID recount logic,
    - category-voting state.

    Writes one row per visible tracked object per frame.
    """

    def __init__(self, path: Path, run_id: str, flush_every: int = 500):
        self.path = path
        self.run_id = run_id
        self.flush_every = flush_every
        self.rows_written = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "run_id",
            "frame",
            "time_sec",

            # Detector/tracker identity
            "track_id",
            "yolo_class",
            "yolo_confidence",

            # Bounding box
            "x1",
            "y1",
            "x2",
            "y2",
            "bbox_w",
            "bbox_h",
            "bbox_area",

            # Reference point and line state
            "reference_point",
            "ref_x",
            "ref_y",
            "line_side_raw",
            "line_distance_px",
            "side_now",
            "prev_side",
            "prev_nonzero_side",

            # Crossing-candidate diagnostics
            "would_cross_using_raw_logic",
            "would_cross_using_nonzero_logic",
            "actual_cross_candidate",
            "direction_candidate",

            # Count suppression / recount diagnostics
            "counted_before",
            "crossed",
            "direction",
            "counted_after",
            "cross_suppressed",
            "suppressed_reason",
            "last_counted_frame",
            "last_counted_direction",
            "frames_since_counted",
            "same_id_recount",
            "allow_same_id_recount",
            "same_id_recount_cooldown",

            # Optional track-lifetime diagnostics
            "track_age_frames",
            "frames_since_last_seen",

            # Classification/voting state
            "current_label",
            "category_confidence",
            "category_scores_json",
            "yolo_summary_json",
        ])

    def write_observation(
        self,
        *,
        frame_idx: int,
        time_sec: float,
        track_id: int,
        yolo_class: int,
        yolo_confidence: float,
        bbox,
        event_debug: dict,
        current_label: str,
        category_confidence: float,
        category_scores: dict | None = None,
        yolo_summary: dict | None = None,
    ):
        x1, y1, x2, y2 = map(float, bbox)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h

        would_cross_raw = bool(event_debug.get("would_cross_using_raw_logic", False))
        would_cross_nonzero = bool(event_debug.get("would_cross_using_nonzero_logic", False))
        actual_cross_candidate = bool(event_debug.get("actual_cross_candidate", False))
        crossed = bool(event_debug.get("crossed", False))

        cross_suppressed = (
            actual_cross_candidate
            and not crossed
        )

        self.writer.writerow([
            self.run_id,
            frame_idx,
            _fmt_float(time_sec, 6),

            int(track_id),
            int(yolo_class),
            _fmt_float(yolo_confidence, 6),

            _fmt_float(x1, 3),
            _fmt_float(y1, 3),
            _fmt_float(x2, 3),
            _fmt_float(y2, 3),
            _fmt_float(bbox_w, 3),
            _fmt_float(bbox_h, 3),
            _fmt_float(bbox_area, 3),

            event_debug.get("reference_point", ""),
            _fmt_float(event_debug.get("ref_x"), 3),
            _fmt_float(event_debug.get("ref_y"), 3),
            _fmt_float(event_debug.get("line_side_raw"), 6),
            _fmt_float(event_debug.get("line_distance_px"), 6),
            _fmt_int(event_debug.get("side_now")),
            _fmt_int(event_debug.get("prev_side")),
            _fmt_int(event_debug.get("prev_nonzero_side")),

            _fmt_bool(would_cross_raw),
            _fmt_bool(would_cross_nonzero),
            _fmt_bool(actual_cross_candidate),
            event_debug.get("direction_candidate", ""),

            _fmt_bool(event_debug.get("counted_before", False)),
            _fmt_bool(crossed),
            event_debug.get("direction", ""),
            _fmt_bool(event_debug.get("counted_after", False)),
            _fmt_bool(cross_suppressed),
            event_debug.get("suppressed_reason", ""),
            _fmt_int(event_debug.get("last_counted_frame")),
            event_debug.get("last_counted_direction", ""),
            _fmt_int(event_debug.get("frames_since_counted")),
            _fmt_bool(event_debug.get("same_id_recount", False)),
            _fmt_bool(event_debug.get("allow_same_id_recount", False)),
            _fmt_int(event_debug.get("same_id_recount_cooldown")),

            _fmt_int(event_debug.get("track_age_frames")),
            _fmt_int(event_debug.get("frames_since_last_seen")),

            current_label,
            _fmt_float(category_confidence, 6),
            _json_safe_dump(category_scores),
            _json_safe_dump(yolo_summary),
        ])

        self.rows_written += 1
        if self.flush_every > 0 and self.rows_written % self.flush_every == 0:
            self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()