"""Nearest-neighbor recognition for the collected fixed-length gesture sequences."""

import json
import math
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "dataset" / "gestures"
METADATA_PATH = DATASET_DIRECTORY / "metadata.json"
MAX_DISTANCE = 20.0
MIN_CLASS_MARGIN = 0.03


class RecognitionRejected(ValueError):
    """Carry diagnostic scoring details for a rejected sequence."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _normalized_hand(hand):
    """Represent a hand relative to its wrist so camera position is ignored."""
    if not hand["present"]:
        return [0.0] + [0.0] * 63

    landmarks = [[float(value) for value in point] for point in hand["landmarks"]]
    wrist = landmarks[0]
    relative = [[value - wrist[index] for index, value in enumerate(point)] for point in landmarks]
    scale = max(
        math.sqrt(sum(value * value for value in point)) for point in relative[1:]
    ) or 1.0
    return [1.0] + [value / scale for point in relative for value in point]


def _frame_vector(frame):
    """Flatten the two fixed hand slots while retaining presence information."""
    values = []
    for hand_name in ("left", "right"):
        values.extend(_normalized_hand(frame["hands"][hand_name]))
    return values


def sequence_vector(sequence):
    """Convert a stored or API sequence into one fixed-size numeric vector."""
    frames = sequence.get("frames", sequence)
    return [value for frame in frames for value in _frame_vector(frame)]


def _distance(first, second):
    if len(first) != len(second):
        return math.inf
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _load_samples(labels):
    samples = []
    for label in labels:
        for sample_path in sorted((DATASET_DIRECTORY / label).glob("sample_*.json")):
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            samples.append((label, sequence_vector(sample)))
    return samples


class GestureRecognizer:
    """Recognize a fixed-length sequence against all labels in metadata.json."""

    def __init__(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.frames_per_sequence = metadata["frames_per_sequence"]
        self.minimum_hand_frames = self.frames_per_sequence - metadata.get(
            "maximum_missing_hand_frames", 0
        )
        self.samples = _load_samples(metadata["labels"])

    def predict(self, sequence):
        frames = sequence.get("frames", sequence)
        if len(frames) != self.frames_per_sequence:
            raise ValueError(
                f"Expected {self.frames_per_sequence} frames, received {len(frames)}."
            )

        detected_frames = sum(
            any(hand["present"] for hand in frame["hands"].values())
            for frame in frames
        )
        if detected_frames < self.minimum_hand_frames:
            diagnostics = {
                "valid_hand_frames": detected_frames,
                "minimum_hand_frames": self.minimum_hand_frames,
                "rejection_reason": "insufficient_hand_frames",
            }
            raise RecognitionRejected(
                f"A hand must be detected in at least {self.minimum_hand_frames} frames.",
                diagnostics,
            )

        vector = sequence_vector(sequence)
        distances_by_label = {}
        for label, sample in self.samples:
            distance = _distance(vector, sample)
            if distance != math.inf:
                distances_by_label.setdefault(label, []).append(distance)
        if not distances_by_label:
            raise ValueError("The gesture sequence does not match the dataset format.")

        class_scores = sorted(
            (
                sum(sorted(distances)[: min(3, len(distances))])
                / min(3, len(distances)),
                label,
            )
            for label, distances in distances_by_label.items()
        )
        best_distance, label = class_scores[0]
        second_best_distance, second_best_label = class_scores[1] if len(class_scores) > 1 else (math.inf, None)
        margin = second_best_distance - best_distance
        diagnostics = {
            "best_label": label,
            "best_distance": round(best_distance, 6),
            "second_best_label": second_best_label,
            "second_best_distance": round(second_best_distance, 6),
            "margin": round(margin, 6),
            "max_distance": MAX_DISTANCE,
            "min_class_margin": MIN_CLASS_MARGIN,
        }
        if best_distance > MAX_DISTANCE or margin < MIN_CLASS_MARGIN:
            reasons = []
            if best_distance > MAX_DISTANCE:
                reasons.append(f"distance>{MAX_DISTANCE}")
            if margin < MIN_CLASS_MARGIN:
                reasons.append(f"margin<{MIN_CLASS_MARGIN}")
            diagnostics["rejection_reason"] = ", ".join(reasons)
            raise RecognitionRejected(
                f"Sign not recognized (distance={best_distance:.6f}, margin={margin:.6f}).",
                diagnostics,
            )

        return {
            "label": label,
            "distance": round(best_distance, 6),
            "second_best_label": second_best_label,
            "second_best_distance": round(second_best_distance, 6),
            "margin": round(margin, 6),
            "samples_considered": sum(len(distances) for distances in distances_by_label.values()),
            "diagnostics": diagnostics,
        }