"""Manually save one-hand MediaPipe landmark samples to dataset/landmarks.csv."""

import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp


LABELS = ["YES", "NO", "HELLO", "THANK_YOU", "QUESTION"]
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
CSV_PATH = PROJECT_DIR / "dataset" / "landmarks.csv"
CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def header():
    columns = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
    return columns + ["label"]


def existing_counts():
    counts = {label: 0 for label in LABELS}
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("label") in counts:
                    counts[row["label"]] += 1
    return counts


def save_sample(landmarks, label):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    values = [coordinate for landmark in landmarks for coordinate in (landmark.x, landmark.y, landmark.z)]
    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header())
        writer.writerow(values + [label])


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for connection in CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the default webcam.")
        return

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    selected_index = 0
    last_save_time = 0.0
    confirmation = "Select a label, show one hand, then press SPACE."

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(image, int(time.monotonic() * 1000))
                landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
                if landmarks:
                    draw_hand(frame, landmarks)

                counts = existing_counts()
                label = LABELS[selected_index]
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), (20, 20, 20), -1)
                cv2.putText(frame, f"Label: {label}   Saved: {counts[label]}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, confirmation, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
                cv2.putText(frame, "1 YES  2 NO  3 HELLO  4 THANK_YOU  5 QUESTION", (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(frame, "SPACE save sample | Q quit", (15, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Silent Meeting AI - Landmark Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if ord("1") <= key <= ord("5"):
                    selected_index = key - ord("1")
                    confirmation = f"Selected {LABELS[selected_index]}. Press SPACE to save one sample."
                elif key == ord(" "):
                    if not landmarks:
                        confirmation = "No hand detected. Sample was not saved."
                    elif time.monotonic() - last_save_time < 0.4:
                        confirmation = "Please release SPACE before saving again."
                    else:
                        save_sample(landmarks, label)
                        last_save_time = time.monotonic()
                        confirmation = f"Saved {label} sample #{counts[label] + 1}."
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
