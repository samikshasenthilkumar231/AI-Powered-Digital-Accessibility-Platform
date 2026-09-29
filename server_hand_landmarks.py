"""Extract the dataset-compatible hand landmarks from uploaded video frames."""

import time

import cv2
import mediapipe as mp
import numpy as np


def _empty_hand():
    return {
        "present": False,
        "handedness": None,
        "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)],
    }


def result_to_frame_data(result, frame_index):
    """Convert one MediaPipe result into the collector's fixed hand slots."""
    hands = {"left": _empty_hand(), "right": _empty_hand()}

    for hand_index, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[hand_index][0].category_name.lower()
        slot = handedness if handedness in hands else "left"
        if hands[slot]["present"]:
            slot = "right" if slot == "left" else "left"
        hands[slot] = {
            "present": True,
            "handedness": handedness,
            "landmarks": [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
        }

    return {
        "frame_index": frame_index,
        "timestamp_ms": int(time.monotonic() * 1000),
        "hands": hands,
    }


class HandLandmarkExtractor:
    """Run MediaPipe Hand Landmarker in video mode for uploaded frames."""

    def __init__(self, model_path):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract(self, image_bytes, frame_index):
        """Decode one JPEG frame and return its fixed-format landmark data."""
        image_array = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_array is None:
            raise ValueError("Unable to decode uploaded camera frame.")

        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.landmarker.detect_for_video(
            mp_image, int(time.monotonic() * 1000)
        )
        return result_to_frame_data(result, frame_index)

    def close(self):
        self.landmarker.close()