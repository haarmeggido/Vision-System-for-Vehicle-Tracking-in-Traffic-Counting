from collections import Counter, defaultdict
import numpy as np

from traffic_counter.config import GDDKIA_LABELS
from traffic_counter.classification.compatibility import get_compatibility_vector

class MajorityVoteBuffer:
    """
    Baseline voting mechanism.

    Classifies every N observed frames per track and stores integer votes.
    Final/current label is the most frequent predicted class.
    """

    name = "majority"

    def __init__(self, classifier, classify_interval: int = 5):
        self.classifier = classifier
        self.classify_interval = max(1, classify_interval)

        self.track_votes = defaultdict(Counter)
        self.track_frame_count = defaultdict(int)

    def observe_crop(self, track_id: int, crop, **kwargs) -> str:
        if crop.size == 0:
            return self.current_label(track_id)

        self.track_frame_count[track_id] += 1

        if (self.track_frame_count[track_id] - 1) % self.classify_interval == 0:
            pred_idx = self.classifier.predict_index(crop)
            self.track_votes[track_id][pred_idx] += 1

        return self.current_label(track_id)

    def current_label(self, track_id: int) -> str:
        if track_id not in self.track_votes:
            return ""

        if len(self.track_votes[track_id]) == 0:
            return ""

        most_common_idx = self.track_votes[track_id].most_common(1)[0][0]
        return GDDKIA_LABELS[most_common_idx]

    def current_confidence(self, track_id: int) -> float:
        """
        Simple confidence proxy for majority voting:
        top vote count / total votes.
        """
        if track_id not in self.track_votes or len(self.track_votes[track_id]) == 0:
            return 0.0

        votes = self.track_votes[track_id]
        total = sum(votes.values())
        if total == 0:
            return 0.0

        return votes.most_common(1)[0][1] / total

    def current_scores(self, track_id: int) -> dict:
        if track_id not in self.track_votes:
            return {}

        votes = self.track_votes[track_id]
        total = sum(votes.values())
        if total == 0:
            return {}

        return {
            GDDKIA_LABELS[idx]: count / total
            for idx, count in votes.items()
        }

    def clear_track_state(self, track_id: int):
        self.track_votes.pop(track_id, None)
        self.track_frame_count.pop(track_id, None)

class ProbabilityVoteBuffer:
    """
    Probability-weighted voting.

    Instead of adding +1 to the top class, this accumulates the full softmax
    probability vector for every sampled crop. Final/current label is the class with the highest accumulated probability.
    """

    name = "probability"

    def __init__(self, classifier, classify_interval: int = 5):
        self.classifier = classifier
        self.classify_interval = max(1, classify_interval)

        self.track_scores = defaultdict(lambda: np.zeros(len(GDDKIA_LABELS), dtype=np.float64))
        self.track_frame_count = defaultdict(int)
        self.track_sample_count = defaultdict(int)

    def observe_crop(self, track_id: int, crop, **kwargs) -> str:
        if crop.size == 0:
            return self.current_label(track_id)

        self.track_frame_count[track_id] += 1

        if (self.track_frame_count[track_id] - 1) % self.classify_interval == 0:
            probs = self.classifier.predict_proba(crop).numpy()

            # First version: pure probability accumulation.
            # Later, this is where crop quality / YOLO compatibility weights will be added.
            self.track_scores[track_id] += probs
            self.track_sample_count[track_id] += 1

        return self.current_label(track_id)

    def current_label(self, track_id: int) -> str:
        if track_id not in self.track_scores:
            return ""

        if self.track_sample_count[track_id] == 0:
            return ""

        best_idx = int(np.argmax(self.track_scores[track_id]))
        return GDDKIA_LABELS[best_idx]

    def current_confidence(self, track_id: int) -> float:
        """
        Normalized confidence proxy:
        top accumulated score / sum of accumulated scores.
        """
        if track_id not in self.track_scores:
            return 0.0

        scores = self.track_scores[track_id]
        total = float(scores.sum())
        if total <= 0:
            return 0.0

        return float(scores.max() / total)

    def current_scores(self, track_id: int) -> dict:
        if track_id not in self.track_scores:
            return {}

        scores = self.track_scores[track_id]
        total = float(scores.sum())
        if total <= 0:
            return {}

        return {
            label: float(score / total)
            for label, score in zip(GDDKIA_LABELS, scores)
        }

    def clear_track_state(self, track_id: int):
        self.track_scores.pop(track_id, None)
        self.track_frame_count.pop(track_id, None)
        self.track_sample_count.pop(track_id, None)


class ProbabilityYoloPriorVoteBuffer(ProbabilityVoteBuffer):
    """
    Probability-weighted voting with a soft YOLO compatibility prior.

    For every sampled crop:
        classifier softmax vector
        × YOLO/GDDKiA compatibility vector
        = evidence added to the track-level score.

    This is not a hard consistency check. It only weakens categories that are
    unlikely according to YOLO's coarse class.
    """

    name = "probability_yolo_prior"

    def __init__(
        self,
        classifier,
        classify_interval: int = 5,
        use_yolo_confidence: bool = False,
    ):
        super().__init__(
            classifier=classifier,
            classify_interval=classify_interval,
        )
        self.use_yolo_confidence = use_yolo_confidence
        self.track_yolo_votes = defaultdict(Counter)

    def observe_crop(
        self,
        track_id: int,
        crop,
        *,
        yolo_class: int | None = None,
        yolo_confidence: float | None = None,
        **kwargs,
    ) -> str:
        if crop.size == 0:
            return self.current_label(track_id)

        self.track_frame_count[track_id] += 1

        # Track YOLO class observations for diagnostics.
        if yolo_class is not None:
            self.track_yolo_votes[track_id][int(yolo_class)] += 1

        if (self.track_frame_count[track_id] - 1) % self.classify_interval == 0:
            probs = self.classifier.predict_proba(crop).numpy()

            compatibility = get_compatibility_vector(yolo_class)
            weighted_probs = probs * compatibility

            # Optional. Keep disabled initially so we can isolate the effect of
            # compatibility prior itself.
            if self.use_yolo_confidence and yolo_confidence is not None:
                weighted_probs = weighted_probs * float(yolo_confidence)

            self.track_scores[track_id] += weighted_probs
            self.track_sample_count[track_id] += 1

        return self.current_label(track_id)

    def current_yolo_summary(self, track_id: int) -> dict:
        if track_id not in self.track_yolo_votes:
            return {}

        return dict(self.track_yolo_votes[track_id])

    def clear_track_state(self, track_id: int):
        super().clear_track_state(track_id)
        self.track_yolo_votes.pop(track_id, None)


def create_vote_buffer(name: str, classifier, classify_interval: int):
    name = name.lower().strip()

    if name == "majority":
        return MajorityVoteBuffer(
            classifier=classifier,
            classify_interval=classify_interval,
        )

    if name in {"probability", "prob", "softmax"}:
        return ProbabilityVoteBuffer(
            classifier=classifier,
            classify_interval=classify_interval,
        )

    if name == "probability_yolo_prior":
        return ProbabilityYoloPriorVoteBuffer(
            classifier=classifier,
            classify_interval=classify_interval,
        )

    raise ValueError(f"Unsupported voting strategy: {name}")