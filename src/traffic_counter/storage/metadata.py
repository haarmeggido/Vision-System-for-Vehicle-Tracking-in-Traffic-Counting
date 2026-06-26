import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import cv2
import torch


def _json_safe(value: Any):
    """
    Convert common non-JSON-safe Python objects into JSON-safe values.
    Handles Path, tuples, dataclasses and nested structures.
    """
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return value


def get_git_commit() -> str | None:
    """
    Returns current git commit hash if available.
    Does not fail if the project is not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_git_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None

def get_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None

def build_run_metadata(
    *,
    run_id: str,
    config,
    video_out_path: Path | None,
    csv_out_path: Path,
    observation_csv_path: Path | None,
    fps: float,
    width: int,
    height: int,
    frames_processed: int,
    elapsed_seconds: float,
    video_seconds: float,
    real_time_factor: float,
    event_counts: dict,
    device: str,
):
    metadata = {
        "run": {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            "git_dirty": get_git_dirty(),
            "command": " ".join(sys.argv),
        },
        "input": {
            "video": config.video,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "outputs": {
            "annotated_video": video_out_path if config.save_video else None,
            "summary_csv": csv_out_path,
            "observation_csv": observation_csv_path,
            "save_video": config.save_video,
            "save_observations": config.save_observations,
        },
        "models": {
            "detector": {
                "type": "ultralytics_yolo",
                "weights": config.yolo_model,
                "classes": list(config.yolo_classes),
                "confidence_threshold": config.conf,
                "iou_threshold": config.iou,
            },
            "tracker": {
                "type": "ultralytics_track",
                "config": config.tracker,
            },
            "classifier": {
                "type": "torchvision",
                "architecture": config.classifier_arch,
                "weights": config.classifier_weights,
                "labels": [
                    "a_bikes",
                    "b_moto",
                    "c_pass",
                    "d_light_comm",
                    "e_heavy_rigid",
                    "f_articulated",
                    "g_bus",
                    "h_agri",
                ],
            },
        },
        "processing": {
            "line": list(config.line),
            "stop_frame": config.stop_frame,
            "classify_interval": config.classify_interval,
            "voting_method": config.voting_method,
            "line_reference": config.line_reference,
            "frames_processed": frames_processed,
            "elapsed_seconds": elapsed_seconds,
            "video_seconds": video_seconds,
            "real_time_factor": real_time_factor,
            "processed_fps": frames_processed / elapsed_seconds if elapsed_seconds > 0 else None,
            "video_fps": fps,
            "events_A_to_B": event_counts.get("A->B", 0),
            "events_B_to_A": event_counts.get("B->A", 0),
            "events_total": event_counts.get("A->B", 0) + event_counts.get("B->A", 0),
            "allow_same_id_recount": config.allow_same_id_recount,
            "same_id_recount_cooldown": config.same_id_recount_cooldown,
            "same_id_recount_events": event_counts.get("same_id_recount_events", 0),
            "suppress_duplicate_events": config.suppress_duplicate_events,
            "duplicate_event_iou_threshold": config.duplicate_event_iou_threshold,
            "duplicate_event_frame_window": config.duplicate_event_frame_window,
            "duplicate_event_center_distance": config.duplicate_event_center_distance,
            "duplicate_events_suppressed": event_counts.get("duplicate_events_suppressed", 0),
        },

        "environment": {
            "device": device,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "opencv_version": cv2.__version__,
        },
    }

    return _json_safe(metadata)


def save_run_metadata(metadata: dict, metadata_path: Path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)