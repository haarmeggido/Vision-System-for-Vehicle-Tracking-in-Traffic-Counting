from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


GDDKIA_LABELS = [
    "a_bikes",
    "b_moto",
    "c_pass",
    "d_light_comm",
    "e_heavy_rigid",
    "f_articulated",
    "g_bus",
    "h_agri",
]

DEFAULT_YOLO_CLASSES = [1, 2, 3, 5, 7]  # COCO: bicycle, car, motorcycle, bus, truck


SUPPORTED_CLASSIFIER_ARCHS = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "swin_t",
    "swin_s",
    "resnet50",
    "resnet101",
    "regnet_y_8gf",
    "densenet121",
    "mobilenet_v3_large",
]

@dataclass(frozen=True)
class PipelineConfig:
    video: Path
    yolo_model: Path
    classifier_weights: Path
    tracker: Path | str
    yolo_classes: Sequence[int]
    conf: float
    iou: float
    line: Tuple[int, int, int, int]
    out_dir: Path
    stop_frame: int
    classify_interval: int
    classifier_arch: str = "convnext_small"
    save_video: bool = False
    voting_method: str = "majority"
    line_reference: str = "centroid"
    save_observations: bool = False
    
    allow_same_id_recount: bool = False
    same_id_recount_cooldown: int = 30
    
    suppress_duplicate_events: bool = False
    duplicate_event_iou_threshold: float = 0.85
    duplicate_event_frame_window: int = 2
    duplicate_event_center_distance: float = 30.0