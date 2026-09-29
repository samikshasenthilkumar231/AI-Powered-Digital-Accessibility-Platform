const video = document.getElementById("webcam");
const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");
const status = document.getElementById("camera-status");
const videoPlaceholder = document.getElementById("video-placeholder");
const speechCaption = document.getElementById("speech-caption");
const startListeningButton = document.getElementById("start-listening");
const stopListeningButton = document.getElementById("stop-listening");
const listeningStatus = document.getElementById("listening-status");
const listeningHelp = document.getElementById("listening-help");
const signStatus = document.getElementById("sign-status");
const signOutput = document.getElementById("sign-output");
const recognitionHelp = document.getElementById("recognition-help");
const FRAME_INTERVAL_MS = 1000 / 30;
const REQUIRED_STABLE_PREDICTIONS = 2;
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");

let cameraStream = null;
let recognition = null;
let isListening = false;
let finalTranscript = "";
let recognitionError = false;
let isRecognizing = false;
let isRecognizingBatch = false;
let recognitionFrames = [];
let recognitionFrameRequest = null;
let stablePrediction = { label: "", count: 0 };
let lastCapturedAt = 0;
let acceptedLabel = "";
let recognitionSession = 0;

function updateSignStatus(message, connected = false) {
  signStatus.classList.toggle("connected", connected);
  signStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${message}`;
}

function resetStablePrediction() {
  stablePrediction = { label: "", count: 0 };
}

function setRecognitionButtons(recognizing) {
  return;
}

function applyStablePrediction(label) {
  if (stablePrediction.label === label) {
    stablePrediction.count += 1;
  } else {
    stablePrediction = { label, count: 1 };
  }

  const normalized = label.replaceAll("_", " ");
  const displayLabel = `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}`;
  if (stablePrediction.count < REQUIRED_STABLE_PREDICTIONS) {
    signOutput.textContent = `Confirming ${displayLabel}...`;
    updateSignStatus("Confirming gesture...", true);
    recognitionHelp.textContent = "Confirming a stable sign...";
    return;
  }

  acceptedLabel = label;
  signOutput.textContent = displayLabel;
  updateSignStatus("Gesture detected", true);
  recognitionHelp.textContent = "Watching for a sign...";
}

async function recognizeSequence(frames, session) {
  updateSignStatus("Recognizing...", true);
  recognitionHelp.textContent = "Recognizing sign...";
  const encodedFrames = await Promise.all(
    frames.map((frame) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Unable to encode camera frame."));
      reader.readAsDataURL(frame);
    }))
  );
  console.log("Sign recognition: sending frame batch", {
    frameCount: encodedFrames.length,
    frameSizes: frames.map((frame) => frame?.size || 0),
  });
  const response = await fetch("/api/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames: encodedFrames }),
  });
  if (!isRecognizing || session !== recognitionSession) {
    return;
  }
  const responseText = await response.text();
  console.log("Sign recognition: API response", {
    status: response.status,
    ok: response.ok,
    body: responseText,
  });
  let result;
  try {
    result = JSON.parse(responseText);
  } catch (error) {
    console.error("Sign recognition: API returned non-JSON response", error);
    throw new Error(`Recognition API returned HTTP ${response.status}.`);
  }
  if (result.diagnostics) {
    console.info("Sign recognition diagnostics", result.diagnostics);
  }
  if (!response.ok) {
    const error = new Error(result.error || "Recognition failed.");
    error.code = result.code;
    throw error;
  }

  const label = result.label.replaceAll("_", " ");
  signOutput.textContent = `✨ Recognizing...`;
  updateSignStatus("Recognizing...", true);
  applyStablePrediction(label);
}

function startRecognitionBatch() {
  if (!isRecognizing || isRecognizingBatch || recognitionFrames.length !== 30) {
    return;
  }

  const frames = recognitionFrames;
  const session = recognitionSession;
  recognitionFrames = [];
  isRecognizingBatch = true;
  console.log("Sign recognition: 30-frame batch complete");
  recognizeSequence(frames, session)
    .catch((error) => {
      if (!isRecognizing || session !== recognitionSession) {
        return;
      }
      console.error("Sign recognition: batch failed", error);
      resetStablePrediction();
      if (!acceptedLabel) {
        signOutput.textContent = error.code === "no_hand_detected" || error.code === "low_confidence"
          ? "👋 Show a sign..."
          : "Recognition unavailable";
      }
      updateSignStatus("Watching for a sign...", true);
      recognitionHelp.textContent = "Watching for a sign...";
    })
    .finally(() => {
      isRecognizingBatch = false;
      if (isRecognizing && session === recognitionSession) {
        startRecognitionBatch();
      }
    });
}

async function recognitionLoop() {
  if (!isRecognizing || !video.srcObject) {
    return;
  }

  const timestamp = performance.now();
  if (timestamp - lastCapturedAt < FRAME_INTERVAL_MS) {
    recognitionFrameRequest = window.requestAnimationFrame(recognitionLoop);
    return;
  }

  // getUserMedia resolves before the first decoded video frame is always ready.
  // Keep the single loop alive until the camera can provide real frames.
  if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    recognitionFrameRequest = window.requestAnimationFrame(recognitionLoop);
    return;
  }

  if (captureCanvas.width !== video.videoWidth || captureCanvas.height !== video.videoHeight) {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
  }
  // Dataset collection mirrors the camera image before MediaPipe assigns
  // handedness. Apply the same transform to live recognition frames.
  captureContext.setTransform(-1, 0, 0, 1, captureCanvas.width, 0);
  captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  const frame = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.8));
  if (!isRecognizing || !video.srcObject) {
    return;
  }
  if (!frame) {
    recognitionFrameRequest = window.requestAnimationFrame(recognitionLoop);
    return;
  }
  lastCapturedAt = timestamp;
  if (recognitionFrames.length === 30) {
    recognitionFrames.shift();
  }
  recognitionFrames.push(frame);
  recognitionHelp.textContent = "Watching for a sign...";

  startRecognitionBatch();

  recognitionFrameRequest = window.requestAnimationFrame(recognitionLoop);
}

async function startSignRecognition() {
  if (!cameraStream || isRecognizing) {
    return;
  }
  try {
    isRecognizing = true;
    recognitionSession += 1;
    recognitionFrames = [];
    resetStablePrediction();
    acceptedLabel = "";
    lastCapturedAt = 0;
    setRecognitionButtons(true);
    updateSignStatus("Watching for a sign...", true);
    signOutput.textContent = "👋 Detecting sign...";
    recognitionHelp.textContent = "Watching for a sign...";
    recognitionFrameRequest = window.requestAnimationFrame(recognitionLoop);
  } catch (error) {
    signOutput.textContent = "Recognition unavailable";
    updateSignStatus("Hand model could not load");
    recognitionHelp.textContent = error.message;
  }
}

function stopRecognition() {
  isRecognizing = false;
  recognitionSession += 1;
  recognitionFrames = [];
  resetStablePrediction();
  acceptedLabel = "";
  lastCapturedAt = 0;
  if (recognitionFrameRequest) {
    window.cancelAnimationFrame(recognitionFrameRequest);
    recognitionFrameRequest = null;
  }
  setRecognitionButtons(false);
  signOutput.textContent = "📷 Start your camera to detect signs";
  updateSignStatus("Recognition ready");
  recognitionHelp.textContent = "📷 Start your camera to detect signs";
}

function updateCameraStatus(message, connected = false) {
  status.classList.toggle("connected", connected);
  status.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${message}`;
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    updateCameraStatus("Camera is not supported in this browser");
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = cameraStream;
    video.classList.add("is-visible");
    videoPlaceholder.classList.add("is-hidden");
    updateCameraStatus("Camera connected", true);
    startButton.disabled = true;
    stopButton.disabled = false;
    setRecognitionButtons(false);
    signOutput.textContent = "👋 Detecting sign...";
    acceptedLabel = "";
    recognitionHelp.textContent = "Watching for a sign...";
    startSignRecognition();
  } catch (error) {
    updateCameraStatus("Camera access was not granted");
    console.error("Unable to access camera:", error);
  }
}

function stopCamera() {
  stopRecognition();
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  video.srcObject = null;
  video.classList.remove("is-visible");
  videoPlaceholder.classList.remove("is-hidden");
  updateCameraStatus("Camera not connected");
  startButton.disabled = false;
  stopButton.disabled = true;
  signOutput.textContent = "📷 Start your camera to detect signs";
  recognitionHelp.textContent = "📷 Start your camera to detect signs";
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

function updateListeningStatus(message, listening = false) {
  listeningStatus.classList.toggle("connected", listening);
  listeningStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${message}`;
}

function updateListeningHelp(message) {
  listeningHelp.textContent = message;
}

function showSpeechCaption(interimTranscript = "") {
  const transcript = `${finalTranscript} ${interimTranscript}`.trim();
  speechCaption.textContent = transcript || "Waiting for speaker...";
}

function setListeningButtons(listening) {
  startListeningButton.disabled = listening;
  stopListeningButton.disabled = !listening;
}

function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const isSupported = Boolean(SpeechRecognition);

  console.log("Speech recognition supported:", isSupported);
  console.log("Browser secure context:", window.isSecureContext);

  if (!SpeechRecognition) {
    updateListeningStatus("Speech Recognition is not supported in this browser");
    startListeningButton.disabled = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    console.log("Speech recognition started");
    recognitionError = false;
    updateListeningStatus("Listening...", true);
    updateListeningHelp("Speak normally. Your live words will appear above.");
    setListeningButtons(true);
  };

  recognition.onresult = (event) => {
    let interimTranscript = "";

    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript;

      if (event.results[index].isFinal) {
        finalTranscript += `${transcript} `;
      } else {
        interimTranscript += transcript;
      }
    }

    showSpeechCaption(interimTranscript);
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);

    if (event.error === "aborted" && !isListening) {
      return;
    }

    if (event.error === "no-speech") {
      updateListeningStatus("Speech recognition error: no-speech", true);
      updateListeningHelp("Try speaking closer to your microphone.");
      return;
    }

    isListening = false;
    recognitionError = true;
    setListeningButtons(false);

    if (event.error === "not-allowed" || event.error === "permission-denied") {
      console.error("Microphone permission denied");
      updateListeningStatus(`Speech recognition error: ${event.error}`);
      updateListeningHelp("Allow microphone access for this site in your browser settings, reload the page, then try again.");
    } else if (event.error === "service-not-allowed") {
      updateListeningStatus("Speech recognition error: service-not-allowed");
      updateListeningHelp("Allow microphone access and make sure your browser permits its speech recognition service.");
    } else if (event.error === "audio-capture") {
      updateListeningStatus("Speech recognition error: audio-capture");
      updateListeningHelp("Connect or enable a microphone in Windows, then try again.");
    } else if (event.error === "network") {
      updateListeningStatus("Speech recognition error: network");
      updateListeningHelp("Check your internet connection, then try Start Listening again.");
    } else {
      updateListeningStatus(`Speech recognition error: ${event.error}`);
      updateListeningHelp(`Browser message: ${event.error}. Try Start Listening again.`);
    }
  };

  recognition.onend = () => {
    console.log("Speech recognition ended");
    if (isListening) {
      window.setTimeout(() => {
        if (isListening) {
          startSpeechRecognition();
        }
      }, 150);
      return;
    }

    setListeningButtons(false);
    if (!recognitionError) {
      updateListeningStatus("Not listening");
      updateListeningHelp("Click Start Listening, then choose Allow when your browser asks to use the microphone.");
    }
  };
}

function startSpeechRecognition() {
  try {
    console.log("Starting Speech Recognition");
    recognition.start();
  } catch (error) {
    const errorName = error.name || "UnknownError";
    isListening = false;
    setListeningButtons(false);
    updateListeningStatus(`Speech recognition start error: ${errorName}`);
    updateListeningHelp(`${errorName}: ${error.message || "No additional browser details."}`);
    console.error("Unable to start speech recognition:", error);
  }
}

async function startListening() {
  if (!recognition || isListening) {
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    updateListeningStatus("Microphone access is not supported in this browser");
    updateListeningHelp("Use a current version of Chrome or another browser that supports microphone access.");
    return;
  }

  isListening = true;
  recognitionError = false;
  setListeningButtons(true);
  updateListeningStatus("Requesting microphone permission...");
  updateListeningHelp("Choose Allow in the browser permission prompt to begin speech captions.");
  console.log("Requesting microphone permission");

  try {
    const permissionStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    console.log("Microphone permission granted");
    permissionStream.getTracks().forEach((track) => track.stop());

    if (isListening) {
      startSpeechRecognition();
    }
  } catch (error) {
    const errorName = error.name || "UnknownError";
    isListening = false;
    recognitionError = true;
    setListeningButtons(false);
    updateListeningStatus(`Microphone error: ${errorName}`);
    updateListeningHelp(`${errorName}: ${error.message || "No additional browser details."} Allow microphone access for this site, reload the page, then try again.`);
    console.error("Microphone permission request failed:", error);
  }
}

function stopListening() {
  if (!recognition) {
    return;
  }

  isListening = false;
  recognitionError = false;
  console.log("Stop Listening button clicked");
  updateListeningStatus("Not listening");
  updateListeningHelp("Click Start Listening when you are ready to capture speech.");

  try {
    recognition.stop();
  } catch (error) {
    console.error("Unable to stop speech recognition:", error);
  }
}

startListeningButton.addEventListener("click", () => {
  console.log("Start Listening button clicked");
  startListening();
});
stopListeningButton.addEventListener("click", stopListening);
setupSpeechRecognition();
