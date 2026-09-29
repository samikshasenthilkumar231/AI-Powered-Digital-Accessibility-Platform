"""Standalone webcam test for MediaPipe's Hand Landmarker task.

Run with:
    signai-venv/Scripts/python.exe tools/test_hand_landmarks.py
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIRECTORY / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def print_hand_landmarks(hand_landmarks, hand_number, handedness):
    """Print normalized x, y, z coordinates for one detected hand."""
    print(f"Hand {hand_number} ({handedness}) landmarks:")
    for landmark_number, landmark in enumerate(hand_landmarks):
        print(
            f"  {landmark_number:2}: "
            f"x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}"
        )


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw MediaPipe's 21 landmarks and their connections on an OpenCV frame."""
    frame_height, frame_width, _ = frame.shape
    points = [
        (int(landmark.x * frame_width), int(landmark.y * frame_height))
        for landmark in hand_landmarks
    ]

    for connection in HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.is_file():
        print(f"Hand Landmarker model file was not found: {MODEL_PATH}")
        return

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

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as hand_landmarker:
            print("Webcam started. Show your hand to the camera. Press 'q' to quit.")

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam. Stopping the test.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[hand_index][0].category_name
                    print_hand_landmarks(hand_landmarks, hand_index + 1, handedness)
                    draw_hand_landmarks(frame, hand_landmarks)

                cv2.imshow("Silent Meeting AI - Hand Landmark Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released. Hand landmark test closed.")


if __name__ == "__main__":
    main()
