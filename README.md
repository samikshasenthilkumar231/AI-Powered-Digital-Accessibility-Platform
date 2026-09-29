# AI-Powered Digital Accessibility Platform for Hearing-Impaired People

A real-time communication prototype that combines live speech captions with camera-based recognition of a small, defined set of hand gestures. It is intended for learning, demonstration, and further development; it is not a general-purpose sign-language translator.

## Project Overview

Communication can be difficult when hearing and deaf or hard-of-hearing people do not share the same communication method. This project explores a browser-based interface that presents spoken words as captions and recognized hand gestures as text.

The prototype includes a Flask application, a MediaPipe hand-landmark extractor, a dataset-based gesture recognizer, and a web interface. The current gesture vocabulary is `hello`, `yes`, `no`, `thank_you`, `wait`, `repeat`, `help`, and `goodbye`.

## Problem Statement

Many everyday conversations rely on speech or shared sign-language fluency. A lightweight tool that captions speech and recognizes a limited set of useful gestures can help demonstrate more accessible interaction, while also making the limits of a small vocabulary clear.

## Solution

The browser requests camera access and continuously captures frames while the camera is active. The Flask server extracts hand landmarks with MediaPipe and compares each fixed-length sequence with labeled examples in the project dataset. Stable, sufficiently close predictions are shown as text. Speech captions use the browser's Web Speech API when supported.

## Key Features

- Live webcam preview with automatic gesture processing after camera permission is granted.
- Recognition of eight dataset-backed gesture labels.
- Two-hand landmark input with no-hand and uncertain-result rejection.
- Temporal confirmation before a gesture is displayed.
- Optional live speech captions in browsers that implement the Web Speech API.
- Data collection and hand-landmark test utilities under `tools/`.

## How the System Works

```text
Webcam frames in the browser
          |
          v
JPEG frame batches sent to the local Flask API
          |
          v
MediaPipe Hand Landmarker extracts left/right hand landmarks
          |
          v
Nearest-neighbor comparison with labeled gesture sequences
          |
          v
Distance and class-margin rejection, then stable UI confirmation
          |
          v
Recognized gesture displayed as text
```

The speech-caption flow is separate: the browser's speech-recognition API converts microphone speech to captions. Browser support and speech-service privacy practices vary by browser and provider.

## Technology Stack

- Python and Flask for the web application and recognition endpoint.
- MediaPipe Tasks Hand Landmarker for hand detection and landmark extraction.
- NumPy and OpenCV for image and landmark processing.
- HTML, CSS, and JavaScript for the browser interface, webcam capture, and speech captions.
- JSON sequences in `dataset/gestures/` for the current gesture examples.

## Project Structure

```text
.
|-- app.py
|-- gesture_recognizer.py
|-- server_hand_landmarks.py
|-- requirements.txt
|-- models/
|   `-- hand_landmarker.task
|-- dataset/
|   `-- gestures/
|       |-- metadata.json
|       |-- goodbye/       # labeled sequence JSON files
|       |-- hello/
|       |-- help/
|       |-- no/
|       |-- repeat/
|       |-- thank_you/
|       |-- wait/
|       `-- yes/
|-- static/
|   |-- script.js
|   `-- style.css
|-- templates/
|   `-- index.html
`-- tools/
    |-- collect_gesture_data.py
    |-- collect_sign_dataset.py
    `-- test_hand_landmarks.py
```

The dataset currently contains 635 landmark sequences across the eight labels. `model/` is empty; Git does not track empty directories. The required MediaPipe task asset is in `models/`.

## Installation

Python 3.14.7 and the pinned dependencies in `requirements.txt` were verified in the development environment.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

From the project root, with the virtual environment activated:

```bash
python -m flask --app app run --host 127.0.0.1
```

Open <http://127.0.0.1:5000> in a supported browser. Camera and microphone access are normally available on `localhost`; other hosts generally need HTTPS. Do not expose Flask's development server directly to the public internet.

## Use Sign Recognition

1. Open the app in a browser and select **Start Camera**. This button is needed for the browser's camera permission; gesture recognition starts automatically after permission is granted.
2. Allow camera access and hold one of the eight supported gestures in view.
3. The application samples a sequence of 30 frames. A hand must be detected in at least 25 frames, matching the dataset's collection rule.
4. A prediction is displayed only after the same label is returned for two consecutive batches. Uncertain predictions are ignored.
5. Select **Stop Camera** when finished to release the webcam.

Speech captions are optional and are controlled separately with **Start Listening**. They require a browser with Web Speech API support and microphone permission.

## Model and Gesture Recognition

`models/hand_landmarker.task` is the MediaPipe Hand Landmarker asset. It detects hands and extracts landmarks; it does not classify the project's eight signs.

Sign classification is performed by `gesture_recognizer.py` using the examples in `dataset/gestures/`. The recognizer expects 30 frames with fixed left/right hand slots and 21 XYZ landmarks per hand. It normalizes landmarks relative to the wrist and hand scale, then compares the sequence against stored examples using Euclidean distance. The class score averages its three closest examples. Predictions are rejected when the best distance exceeds `20.0` or the margin over the second-best class is below `0.03`. These are distance-based thresholds, not calibrated probabilities.

This dataset-backed prototype recognizes only the listed isolated gestures. It does not translate continuous sign-language sentences, identify fingerspelling, or guarantee accuracy for signs, users, camera positions, or lighting conditions not represented in the data. Improving those capabilities requires broader data and independent evaluation.

## Privacy and Public-Repository Checklist

- The gesture dataset contains hand landmarks rather than saved camera images. During recognition, image frames are sent to the local Flask API for landmark extraction; the application does not save those uploaded frames.
- Confirm that everyone whose gestures contributed to the dataset agreed to public redistribution. Landmark sequences can still be sensitive biometric-like data even without photographs or names.
- Confirm the source and redistribution terms for `models/hand_landmarker.task` before publishing it.
- Speech recognition is provided by the browser; review that browser/provider's microphone and speech-processing practices.
- Keep credentials, local settings, virtual environments, logs, and machine-specific files out of the repository. `.gitignore` covers common local artifacts but is not a substitute for reviewing staged files.

## Future Enhancements

- Collect a larger, more diverse dataset and document consent and provenance.
- Add repeatable train/validation/test splits and report per-class performance.
- Evaluate a purpose-trained temporal classifier and calibrate its rejection thresholds.
- Expand the vocabulary and investigate continuous sign recognition with signer-independent testing.
- Improve accessibility testing across browsers, devices, camera conditions, and assistive technologies.

## Team and Project Information

- **Project:** AI-Powered Digital Accessibility Platform for Hearing-Impaired People
- **Team name:** [Add team name]
- **Contributors:** [Add contributor names]
- **Institution or course:** [Add details, if applicable]
- **Supervisor or mentor:** [Add details, if applicable]

Add only information that contributors have agreed to publish. Confirm the licenses and permissions for the source code, dataset, and bundled model asset before choosing a public repository license.
