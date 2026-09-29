"""Manually collect MediaPipe hand-landmark sequences for the gesture vocabulary.

Run with:
    signai-venv/Scripts/python.exe tools/collect_gesture_data.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp


LABELS = ["hello", "yes", "no", "thank_you", "wait", "repeat", "help", "goodbye"]
FRAMES_PER_SEQUENCE = 30
MAX_MISSING_HAND_FRAMES = 5
TARGET_SAMPLES_PER_LABEL = 20

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIRECTORY / "models" / "hand_landmarker.task"
DATASET_DIRECTORY = PROJECT_DIRECTORY / "dataset" / "gestures"
METADATA_PATH = DATASET_DIRECTORY / "metadata.json"
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def empty_landmarks():
    """Return a zero-padded, fixed-size landmark list for a missing hand."""
    return [[0.0, 0.0, 0.0] for _ in range(21)]


def empty_hand_slot():
    """Return the consistent representation used when a hand is not detected."""
    return {"present": False, "handedness": None, "landmarks": empty_landmarks()}


def create_dataset_structure():
    """Create all label folders and the first metadata file when needed."""
    for label in LABELS:
        (DATASET_DIRECTORY / label).mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        save_metadata()


def sample_count(label):
    """Count saved JSON sequences for one label."""
    return len(list((DATASET_DIRECTORY / label).glob("sample_*.json")))


def save_metadata():
    """Save dataset details and current sample counts."""
    metadata = {
        "labels": LABELS,
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "target_valid_sequences_per_label": TARGET_SAMPLES_PER_LABEL,
        "maximum_missing_hand_frames": MAX_MISSING_HAND_FRAMES,
        "feature_format": {
            "per_frame": "fixed left/right hand slots",
            "per_hand": "present mask, handedness, and 21 normalized [x, y, z] landmarks",
            "missing_hand_representation": "present=false and all 21 landmarks padded with [0.0, 0.0, 0.0]",
            "raw_camera_images_saved": False,
        },
        "collection_date": datetime.now(timezone.utc).isoformat(),
        "samples_per_label": {label: sample_count(label) for label in LABELS},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw the 21 landmarks and MediaPipe hand connections on the preview."""
    frame_height, frame_width, _ = frame.shape
    points = [
        (int(landmark.x * frame_width), int(landmark.y * frame_height))
        for landmark in hand_landmarks
    ]

    for connection in HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def result_to_frame_data(result, frame_index, timestamp_ms):
    """Convert a MediaPipe result into two fixed hand slots for one video frame."""
    hand_slots = {"left": empty_hand_slot(), "right": empty_hand_slot()}

    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[hand_index][0].category_name.lower()
        slot_name = handedness if handedness in hand_slots else "left"

        # If MediaPipe ever returns two hands with the same label, keep both.
        if hand_slots[slot_name]["present"]:
            slot_name = "right" if slot_name == "left" else "left"

        hand_slots[slot_name] = {
            "present": True,
            "handedness": handedness,
            "landmarks": [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        }

    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "hands": hand_slots,
    }


def frame_has_hand(frame_data):
    """Return True if at least one hand was detected in this frame."""
    return any(hand["present"] for hand in frame_data["hands"].values())


def frame_is_valid(frame_data):
    """Return True when a frame has the exact recognition input structure."""
    if not isinstance(frame_data, dict) or not isinstance(frame_data.get("hands"), dict):
        return False
    for hand_name in ("left", "right"):
        hand = frame_data["hands"].get(hand_name)
        if not isinstance(hand, dict) or not isinstance(hand.get("present"), bool):
            return False
        if not isinstance(hand.get("landmarks"), list) or len(hand["landmarks"]) != 21:
            return False
        if any(not isinstance(point, list) or len(point) != 3 for point in hand["landmarks"]):
            return False
    return True


def sequence_signature(sequence_frames):
    """Create a timestamp-free signature for exact duplicate detection."""
    return [
        {
            hand_name: {
                "present": frame["hands"][hand_name]["present"],
                "handedness": frame["hands"][hand_name]["handedness"],
                "landmarks": frame["hands"][hand_name]["landmarks"],
            }
            for hand_name in ("left", "right")
        }
        for frame in sequence_frames
    ]


def is_duplicate_sequence(label, sequence_frames):
    """Return True if this label already contains the same landmark sequence."""
    signature = sequence_signature(sequence_frames)
    for sample_path in (DATASET_DIRECTORY / label).glob("sample_*.json"):
        existing = json.loads(sample_path.read_text(encoding="utf-8"))
        if sequence_signature(existing.get("frames", [])) == signature:
            return True
    return False


def save_sequence(label, sequence_frames):
    """Validate and save one complete landmark sequence as JSON."""
    if len(sequence_frames) != FRAMES_PER_SEQUENCE or not all(
        frame_is_valid(frame_data) for frame_data in sequence_frames
    ):
        print("Sequence rejected: landmark frame structure is invalid.")
        return False

    missing_hand_frames = sum(not frame_has_hand(frame_data) for frame_data in sequence_frames)

    if missing_hand_frames > MAX_MISSING_HAND_FRAMES:
        print(
            f"Sequence rejected: {missing_hand_frames}/{FRAMES_PER_SEQUENCE} frames had no detected hand. "
            f"Maximum allowed is {MAX_MISSING_HAND_FRAMES}."
        )
        return False

    if is_duplicate_sequence(label, sequence_frames):
        print(f"Sequence rejected: duplicate {label} sequence.")
        return False

    sequence_number = sample_count(label) + 1
    sample_path = DATASET_DIRECTORY / label / f"sample_{sequence_number:03}.json"
    sample = {
        "label": label,
        "sequence_number": sequence_number,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "frames": sequence_frames,
    }
    sample_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    save_metadata()
    print(f"Saved valid sequence: {sample_path}")
    return True


def draw_interface(frame, label, progress, countdown_seconds, message):
    """Draw label, progress, and keyboard instructions on the webcam preview."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 128), (20, 20, 20), -1)
    cv2.putText(frame, f"Selected label: {label}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"Next sample: {sample_count(label) + 1}/{TARGET_SAMPLES_PER_LABEL}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(frame, f"Frame progress: {progress}/{FRAMES_PER_SEQUENCE}", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, message, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if countdown_seconds is not None:
        cv2.putText(
            frame,
            f"Get ready: {countdown_seconds}",
            (frame.shape[1] // 2 - 120, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

    controls = "1=hello  2=yes  3=no  4=thank_you  5=wait  6=repeat  7=help  8=goodbye"
    controls += "  R=record  Q=quit"
    cv2.putText(frame, controls, (16, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    if not MODEL_PATH.is_file():
        print(f"Hand Landmarker model file was not found: {MODEL_PATH}")
        return

    create_dataset_structure()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the default webcam. Check that it is connected and not in use.")
        return

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    selected_label = "hello"
    countdown_started_at = None
    recording = False
    sequence_frames = []

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as hand_landmarker:
            print("Collection tool started. Select a label with 1-8, press R to record, Q to quit.")

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam. Stopping collection.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.monotonic() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

                countdown_seconds = None
                message = "Choose a label, then press R to record."

                if countdown_started_at is not None:
                    elapsed_seconds = time.monotonic() - countdown_started_at
                    countdown_seconds = max(0, 3 - int(elapsed_seconds))
                    message = "Hold your gesture ready. Recording begins after the countdown."

                    if elapsed_seconds >= 3:
                        countdown_started_at = None
                        recording = True
                        sequence_frames = []
                        countdown_seconds = None
                        message = "Recording exactly 30 frames now. Keep the gesture visible."

                if recording:
                    sequence_frames.append(
                        result_to_frame_data(result, len(sequence_frames), timestamp_ms)
                    )
                    message = "Recording exactly 30 frames now. Keep the gesture visible."

                    if len(sequence_frames) == FRAMES_PER_SEQUENCE:
                        save_sequence(selected_label, sequence_frames)
                        recording = False
                        sequence_frames = []

                draw_interface(
                    frame,
                    selected_label,
                    len(sequence_frames),
                    countdown_seconds,
                    message,
                )
                cv2.imshow("Silent Meeting AI - Gesture Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if not recording and countdown_started_at is None:
                    if ord("1") <= key <= ord("8"):
                        selected_label = LABELS[key - ord("1")]
                        print(f"Selected label: {selected_label}")
                    elif key in (ord("r"), ord("R")):
                        countdown_started_at = time.monotonic()
                        print(f"Get ready to record: {selected_label}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released. Gesture data collection tool closed.")


if __name__ == "__main__":
    main()
