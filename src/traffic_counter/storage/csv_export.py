import csv


class EventCSVWriter:
    """
    Baseline CSV writer.
    """

    def __init__(self, path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["run_id", "frame", "vehicle_id", "gddkia_class", "category_confidence", "direction"])

    def write_event(self, frame_idx: int, track_id: int, label: str, direction: str, category_confidence: float = 0.0, ):
        self.writer.writerow([self.run_id, frame_idx, track_id, label, f"{category_confidence:.6f}", direction])

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()