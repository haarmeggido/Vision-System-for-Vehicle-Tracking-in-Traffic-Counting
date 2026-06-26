from ultralytics import YOLO


class UltralyticsTracker:
    """
    Thin wrapper around Ultralytics YOLO.track().

    This intentionally preserves the current approach where Ultralytics handles
    both detection and tracking internally.
    """

    def __init__(
        self,
        model_path,
        tracker_config,
        classes,
        conf: float = 0.25,
        iou: float = 0.5,
    ):
        self.model = YOLO(str(model_path))
        self.tracker_config = str(tracker_config)
        self.classes = list(classes)
        self.conf = conf
        self.iou = iou

    def stream(self, video_path):
        return self.model.track(
            source=str(video_path),
            classes=self.classes,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            stream=True,
            verbose=False,
        )