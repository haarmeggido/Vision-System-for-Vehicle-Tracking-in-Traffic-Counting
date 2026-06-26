import numpy as np

from traffic_counter.config import GDDKIA_LABELS


# COCO classes used by your YOLO stage:
# 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
#
# These are soft compatibility priors, not hard filters.
# A value below 1.0 reduces the contribution of a class but does not remove it.
YOLO_TO_GDDKIA_COMPATIBILITY = {
    1: {  # bicycle
        "a_bikes": 1.00,
        "b_moto": 0.85,
        "c_pass": 0.20,
        "d_light_comm": 0.15,
        "e_heavy_rigid": 0.10,
        "f_articulated": 0.10,
        "g_bus": 0.10,
        "h_agri": 0.40,
    },
    2: {  # car
        "a_bikes": 0.10,
        "b_moto": 0.10,
        "c_pass": 1.00,
        "d_light_comm": 1.00,
        "e_heavy_rigid": 0.35,
        "f_articulated": 0.25,
        "g_bus": 0.35,
        "h_agri": 0.50,
    },
    3: {  # motorcycle
        "a_bikes": 0.85,
        "b_moto": 1.00,
        "c_pass": 0.15,
        "d_light_comm": 0.10,
        "e_heavy_rigid": 0.10,
        "f_articulated": 0.10,
        "g_bus": 0.10,
        "h_agri": 0.30,
    },
    5: {  # bus
        "a_bikes": 0.10,
        "b_moto": 0.10,
        "c_pass": 0.45,
        "d_light_comm": 0.45,
        "e_heavy_rigid": 0.50,
        "f_articulated": 0.40,
        "g_bus": 1.00,
        "h_agri": 0.40,
    },
    7: {  # truck
        "a_bikes": 0.10,
        "b_moto": 0.10,
        "c_pass": 0.45,
        "d_light_comm": 1.00,
        "e_heavy_rigid": 1.00,
        "f_articulated": 1.00,
        "g_bus": 0.45,
        "h_agri": 0.75,
    },
}


DEFAULT_COMPATIBILITY = 0.50


def get_compatibility_vector(
    yolo_class: int | None,
    default: float = DEFAULT_COMPATIBILITY,
) -> np.ndarray:
    """
    Return a vector of compatibility weights aligned with GDDKIA_LABELS.

    This should be treated as a soft prior:
        1.0  = fully compatible
        0.5  = uncertain / weakly compatible
        0.1  = unlikely, but not impossible

    It intentionally does not use zero values, because YOLO's coarse class may
    be wrong or insufficient for special cases such as h_agri.
    """
    if yolo_class is None:
        return np.ones(len(GDDKIA_LABELS), dtype=np.float64)

    class_map = YOLO_TO_GDDKIA_COMPATIBILITY.get(int(yolo_class))

    if class_map is None:
        return np.full(len(GDDKIA_LABELS), default, dtype=np.float64)

    return np.array(
        [class_map.get(label, default) for label in GDDKIA_LABELS],
        dtype=np.float64,
    )