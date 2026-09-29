import base64
import binascii
import json
from pathlib import Path

import logging

from flask import Flask, jsonify, render_template, request

from gesture_recognizer import METADATA_PATH, GestureRecognizer, RecognitionRejected
from server_hand_landmarks import HandLandmarkExtractor


app = Flask(__name__)
app.logger.setLevel(logging.INFO)
recognizer = None
extractor = None
MODEL_PATH = Path(app.root_path) / "models" / "hand_landmarker.task"
def recognition_diagnostics(frames, scoring=None, rejection_reason=None):
    """Build development diagnostics without changing the user-facing result."""
    detected_frame_count = sum(
        any(hand["present"] for hand in frame["hands"].values())
        for frame in frames
    )
    minimum_frames = (
        recognizer.minimum_hand_frames
        if recognizer is not None
        else json.loads(METADATA_PATH.read_text(encoding="utf-8"))["frames_per_sequence"]
        - json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
            "maximum_missing_hand_frames", 0
        )
    )
    diagnostics = dict(scoring or {})
    diagnostics.update(
        {
            "rejection_threshold": {
                "max_distance": 20.0,
                "min_class_margin": 0.03,
            },
            "hand_detection": {
                "valid_frames": detected_frame_count,
                "total_frames": len(frames),
                "minimum_valid_frames": minimum_frames,
                "enough_frames": detected_frame_count >= minimum_frames,
            },
            "rejection_reason": rejection_reason,
        }
    )
    return diagnostics


@app.route("/")
def index():
    """Display the meeting interface."""
    return render_template("index.html")


@app.post("/api/recognize")
def recognize_gesture():
    """Recognize uploaded camera frames or an existing landmark sequence."""
    global extractor, recognizer

    uploaded_frames = request.files.getlist("frames")
    app.logger.info(
        "recognize request: content_type=%s uploaded_frame_count=%d content_length=%s",
        request.content_type,
        len(uploaded_frames),
        request.content_length,
    )
    if uploaded_frames:
        try:
            if extractor is None:
                app.logger.info("initializing MediaPipe hand landmark extractor")
                extractor = HandLandmarkExtractor(MODEL_PATH)
            frames = [
                extractor.extract(frame.read(), index)
                for index, frame in enumerate(uploaded_frames)
            ]
            detected_frame_count = sum(
                any(hand["present"] for hand in frame["hands"].values())
                for frame in frames
            )
            app.logger.info(
                "landmark extraction complete: frame_count=%d detected_frame_count=%d",
                len(frames),
                detected_frame_count,
            )
            if not any(
                hand["present"]
                for frame in frames
                for hand in frame["hands"].values()
            ):
                return jsonify(
                    {
                        "error": "No hand detected",
                        "code": "no_hand_detected",
                        "diagnostics": recognition_diagnostics(
                            frames, rejection_reason="no_hand_detected"
                        ),
                    }
                ), 422
            payload = {"frames": frames}
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
            app.logger.exception("MediaPipe frame extraction failed")
            return jsonify({"error": str(error)}), 400
    else:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            return jsonify({"error": "Expected uploaded frames or a JSON frames list."}), 400
        app.logger.info(
            "JSON payload frame types: %s",
            [type(frame).__name__ for frame in payload["frames"][:3]],
        )
        if payload["frames"] and any(
            not isinstance(frame, dict) for frame in payload["frames"]
        ):
            try:
                if extractor is None:
                    app.logger.info("initializing MediaPipe hand landmark extractor")
                    extractor = HandLandmarkExtractor(MODEL_PATH)
                frames = []
                for index, data_url in enumerate(payload["frames"]):
                    if not isinstance(data_url, str):
                        raise ValueError("Camera frame payload must be a data URL string.")
                    _, encoded_frame = data_url.split(",", 1)
                    frame_bytes = base64.b64decode(encoded_frame, validate=True)
                    frames.append(extractor.extract(frame_bytes, index))
                detected_frame_count = sum(
                    any(hand["present"] for hand in frame["hands"].values())
                    for frame in frames
                )
                app.logger.info(
                    "JSON frame extraction complete: frame_count=%d detected_frame_count=%d",
                    len(frames),
                    detected_frame_count,
                )
                if not detected_frame_count:
                    return jsonify(
                        {
                            "error": "No hand detected",
                            "code": "no_hand_detected",
                            "diagnostics": recognition_diagnostics(
                                frames, rejection_reason="no_hand_detected"
                            ),
                        }
                    ), 422
                payload = {"frames": frames}
            except (ValueError, binascii.Error) as error:
                app.logger.exception("JSON camera frame extraction failed")
                return jsonify({"error": str(error)}), 400

    try:
        if recognizer is None:
            app.logger.info("initializing gesture recognizer")
            recognizer = GestureRecognizer()
        prediction = recognizer.predict(payload)
        prediction["diagnostics"] = recognition_diagnostics(
            payload["frames"], prediction.get("diagnostics")
        )
        app.logger.info("gesture recognition result: %s", prediction)
    except RecognitionRejected as error:
        diagnostics = recognition_diagnostics(
            payload["frames"], error.diagnostics, error.diagnostics.get("rejection_reason")
        )
        app.logger.info("gesture recognition diagnostics: %s", diagnostics)
        rejection_code = (
            "no_hand_detected"
            if error.diagnostics.get("rejection_reason") == "insufficient_hand_frames"
            else "low_confidence"
        )
        return jsonify(
            {
                "error": str(error),
                "code": rejection_code,
                "diagnostics": diagnostics,
            }
        ), 422
    except (OSError, KeyError, TypeError, ValueError) as error:
        app.logger.exception("gesture recognizer failed")
        if str(error).startswith("Sign not recognized"):
            return jsonify({"error": str(error), "code": "low_confidence"}), 422
        return jsonify({"error": str(error)}), 400

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True)
