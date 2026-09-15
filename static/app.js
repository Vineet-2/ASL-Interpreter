/**
 * Live ASL interpreter client: camera, MediaPipe Hands, WebSocket pipeline.
 */

const CORE_VOCAB = [
  "ME", "YOU", "WE",
  "WANT", "NEED", "HAVE", "LIKE", "GO", "EAT", "DRINK", "WORK", "HELP", "FINISH",
  "HOME", "SCHOOL", "SHOP", "BATHROOM",
  "WATER", "PHONE",
  "TODAY", "TOMORROW", "NOW", "LATER",
  "WHAT", "WHERE", "WHO", "HOW",
  "HELLO", "GOODBYE", "PLEASE", "THANKYOU", "YES", "NO",
  "GOOD", "BAD", "HAPPY", "TIRED",
  "NOT", "MORE"
];

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

let ws = null;
let wsBusy = false;
let activeTokens = [];
let mediaStream = null;
let handsDetector = null;
let isWebcamActive = true;
let isTTSActive = true;
let isProcessingFrame = false;
let animationFrameId = null;
let lastHandsPresent = false;

let frameTimes = [];
let lastFpsUpdate = performance.now();
let lastWsSendTime = 0;
let lastSpokenSentence = "";
let lastTranscriptKey = "";

const wsStatus = document.getElementById("wsStatus");
const videoElement = document.getElementById("webcamVideo");
const canvasElement = document.getElementById("landmarkCanvas");
const canvasCtx = canvasElement.getContext("2d");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const fpsDisplay = document.getElementById("fpsDisplay");
const bufferDisplay = document.getElementById("bufferDisplay");
const handsDisplay = document.getElementById("handsDisplay");
const liveHypothesis = document.getElementById("liveHypothesis");
const confBar = document.getElementById("confBar");
const holdBar = document.getElementById("holdBar");
const holdLabel = document.getElementById("holdLabel");
const sensitivitySelect = document.getElementById("sensitivitySelect");
const phraseChips = document.getElementById("phraseChips");
const backendPill = document.getElementById("backendPill");
const liveSubtitle = document.getElementById("liveSubtitle");
const providerTag = document.getElementById("providerTag");
const latencyTag = document.getElementById("latencyTag");
const totalLatencyBadge = document.getElementById("totalLatencyBadge");
const toggleWebcam = document.getElementById("toggleWebcam");
const camSwitchLabel = document.getElementById("camSwitchLabel");
const toggleTTS = document.getElementById("toggleTTS");
const activeTokensContainer = document.getElementById("activeTokens");
const vocabPallet = document.getElementById("vocabPallet");
const btnClearTokens = document.getElementById("btnClearTokens");
const btnSendSigns = document.getElementById("btnSendSigns");
const transcriptList = document.getElementById("transcriptList");
const btnExportJson = document.getElementById("btnExportJson");
const btnResetSession = document.getElementById("btnResetSession");
const btnCommitPhrase = document.getElementById("btnCommitPhrase");

const tabBtnDashboard = document.getElementById("tabBtnDashboard");
const tabBtnTesting = document.getElementById("tabBtnTesting");
const viewDashboard = document.getElementById("viewDashboard");
const viewTesting = document.getElementById("viewTesting");

const inspectRaw = document.getElementById("inspectRaw");
const inspectDisambiguated = document.getElementById("inspectDisambiguated");
const inspectEnglish = document.getElementById("inspectEnglish");
const inspectSpeech = document.getElementById("inspectSpeech");

const btnRunBenchmark = document.getElementById("btnRunBenchmark");
const benchContextAcc = document.getElementById("benchContextAcc");
const benchNoContextAcc = document.getElementById("benchNoContextAcc");
const benchDelta = document.getElementById("benchDelta");
const benchmarkTableBody = document.getElementById("benchmarkTableBody");

function setupTabs() {
  tabBtnDashboard.addEventListener("click", () => {
    tabBtnDashboard.classList.add("active");
    tabBtnTesting.classList.remove("active");
    viewDashboard.style.display = "";
    viewTesting.style.display = "none";
  });

  tabBtnTesting.addEventListener("click", () => {
    tabBtnTesting.classList.add("active");
    tabBtnDashboard.classList.remove("active");
    viewDashboard.style.display = "none";
    viewTesting.style.display = "grid";
    loadBenchmarkTable();
  });
}

async function startCameraStream() {
  try {
    const constraints = {
      video: { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 24, max: 30 } },
      audio: false
    };
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = mediaStream;
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => { videoElement.play(); resolve(); };
    });
    videoPlaceholder.classList.remove("visible");
    camSwitchLabel.textContent = "Camera on";
    isWebcamActive = true;
    bufferDisplay.textContent = "Tracking";
    runVideoDetectionLoop();
  } catch (err) {
    console.error("Camera access error:", err);
    videoPlaceholder.classList.add("visible");
    camSwitchLabel.textContent = "Camera offline";
    fpsDisplay.textContent = "Error";
    bufferDisplay.textContent = "No cam";
  }
}

function stopCameraStream() {
  isWebcamActive = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  videoElement.srcObject = null;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  videoPlaceholder.classList.add("visible");
  camSwitchLabel.textContent = "Camera off";
  fpsDisplay.textContent = "OFF";
  bufferDisplay.textContent = "Standby";
}

toggleWebcam.addEventListener("change", (e) => {
  if (e.target.checked) startCameraStream();
  else stopCameraStream();
});

function initMediaPipe() {
  if (typeof Hands === "undefined") {
    console.warn("MediaPipe Hands library not loaded.");
    return;
  }
  handsDetector = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  handsDetector.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.55,
    minTrackingConfidence: 0.5
  });
  handsDetector.onResults(onHandResults);
}

function runVideoDetectionLoop() {
  if (!isWebcamActive || !videoElement.videoWidth) {
    if (isWebcamActive) animationFrameId = requestAnimationFrame(runVideoDetectionLoop);
    return;
  }
  const now = performance.now();
  frameTimes.push(now);
  while (frameTimes.length > 0 && frameTimes[0] <= now - 1000) frameTimes.shift();
  if (now - lastFpsUpdate > 300) {
    fpsDisplay.textContent = `${frameTimes.length} FPS`;
    lastFpsUpdate = now;
  }
  if (handsDetector && !isProcessingFrame) {
    isProcessingFrame = true;
    handsDetector.send({ image: videoElement })
      .catch((e) => console.warn("Hands detection skipped frame:", e))
      .finally(() => { isProcessingFrame = false; });
  }
  if (isWebcamActive) animationFrameId = requestAnimationFrame(runVideoDetectionLoop);
}

function packLandmarks(results) {
  const lh = new Array(63).fill(0.0);
  const rh = new Array(63).fill(0.0);
  if (!results.multiHandedness || !results.multiHandLandmarks) return lh.concat(rh);

  results.multiHandedness.forEach((handMeta, idx) => {
    // Front webcam: MediaPipe's Left/Right is mirrored vs the signer. Swap to match Holistic training.
    const isRight = handMeta.label === "Left";
    const landmarks = results.multiHandLandmarks[idx];
    const target = isRight ? rh : lh;
    landmarks.forEach((pt, i) => {
      if (i < 21) {
        target[i * 3] = pt.x;
        target[i * 3 + 1] = pt.y;
        target[i * 3 + 2] = pt.z;
      }
    });
  });
  return lh.concat(rh);
}

function sendLandmarkFrame(featureVector, handsPresent) {
  const now = performance.now();
  const minGap = handsPresent ? 50 : 100;
  if (now - lastWsSendTime < minGap) return;
  if (!ws || ws.readyState !== WebSocket.OPEN || wsBusy) return;
  lastWsSendTime = now;
  wsBusy = true;
  ws.send(JSON.stringify({
    type: "landmarks",
    landmarks: [featureVector],
    hands_present: handsPresent,
    sensitivity: sensitivitySelect ? sensitivitySelect.value : "steady"
  }));
  window.setTimeout(() => { wsBusy = false; }, 300);
}

function onHandResults(results) {
  if (!isWebcamActive) return;
  canvasElement.width = videoElement.videoWidth || 640;
  canvasElement.height = videoElement.videoHeight || 360;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  const handsPresent = !!(results.multiHandLandmarks && results.multiHandLandmarks.length);
  lastHandsPresent = handsPresent;
  if (handsDisplay) handsDisplay.textContent = handsPresent ? `${results.multiHandLandmarks.length} hand(s)` : "No hands";

  if (handsPresent) {
    results.multiHandLandmarks.forEach((landmarks) => {
      canvasCtx.strokeStyle = "rgba(224, 122, 61, 0.85)";
      canvasCtx.lineWidth = 2;
      HAND_CONNECTIONS.forEach(([a, b]) => {
        const pa = landmarks[a];
        const pb = landmarks[b];
        canvasCtx.beginPath();
        canvasCtx.moveTo(pa.x * canvasElement.width, pa.y * canvasElement.height);
        canvasCtx.lineTo(pb.x * canvasElement.width, pb.y * canvasElement.height);
        canvasCtx.stroke();
      });
      landmarks.forEach((pt) => {
        canvasCtx.beginPath();
        canvasCtx.arc(pt.x * canvasElement.width, pt.y * canvasElement.height, 3.2, 0, 2 * Math.PI);
        canvasCtx.fillStyle = "#7eb8a4";
        canvasCtx.fill();
      });
    });
  }

  sendLandmarkFrame(packLandmarks(results), handsPresent);
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    wsBusy = false;
    wsStatus.querySelector(".status-dot").classList.add("connected");
    wsStatus.querySelector(".status-label").textContent = "Live";
    if (sensitivitySelect) {
      ws.send(JSON.stringify({
        type: "sensitivity",
        sensitivity: sensitivitySelect.value
      }));
    }
  };
  ws.onclose = () => {
    wsBusy = false;
    wsStatus.querySelector(".status-dot").classList.remove("connected");
    wsStatus.querySelector(".status-label").textContent = "Reconnecting…";
    setTimeout(connectWebSocket, 2000);
  };
  ws.onerror = (err) => console.error("WebSocket error:", err);
  ws.onmessage = (event) => {
    wsBusy = false;
    try {
      handlePipelineResult(JSON.parse(event.data));
    } catch (e) {
      console.error("Error parsing WebSocket message:", e);
    }
  };
}

if (sensitivitySelect) {
  const saved = localStorage.getItem("asl_sensitivity") || "steady";
  sensitivitySelect.value = saved;
  sensitivitySelect.addEventListener("change", () => {
    localStorage.setItem("asl_sensitivity", sensitivitySelect.value);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "sensitivity",
        sensitivity: sensitivitySelect.value
      }));
    }
  });
}

function renderPhraseChips(glosses) {
  if (!phraseChips) return;
  if (!glosses || !glosses.length) {
    phraseChips.innerHTML = `<span class="muted">Signs you hold will appear here</span>`;
    return;
  }
  phraseChips.innerHTML = glosses.map((g) => `<span class="chip">${g}</span>`).join("");
}

function handlePipelineResult(data) {
  if (!data) return;

  if (backendPill && data.sign_backend) {
    const label = data.sign_backend === "geometric" || data.sign_backend === "geometric_fallback"
      ? "Geometry interpreter (no trained BiGRU)"
      : `Recognizer: ${data.sign_backend}`;
    backendPill.textContent = label;
  }
  if (bufferDisplay && data.buffer_len != null) {
    bufferDisplay.textContent = `Buffer ${data.buffer_len}/30`;
  }

  if (liveHypothesis) {
    liveHypothesis.textContent = data.live_hypothesis || "—";
  }
  if (confBar) {
    const pct = Math.round((data.live_confidence || 0) * 100);
    confBar.style.width = `${pct}%`;
  }
  if (holdBar) {
    const holdPct = Math.round((data.hold_progress || 0) * 100);
    holdBar.style.width = `${holdPct}%`;
  }
  if (holdLabel) {
    if (data.hold_progress >= 0.95) {
      holdLabel.textContent = "Locked ✓";
      holdLabel.style.color = "var(--success)";
    } else if (data.hold_progress > 0) {
      holdLabel.textContent = "Locking in…";
      holdLabel.style.color = "var(--accent)";
    } else {
      holdLabel.textContent = "Hold to commit";
      holdLabel.style.color = "var(--accent-2)";
    }
  }
  if (data.pending_glosses) renderPhraseChips(data.pending_glosses);

  if (data.english_sentence) {
    liveSubtitle.textContent = data.english_sentence;
    providerTag.textContent = `Provider: ${data.translation_provider || data.provider || "Local"}`;
    latencyTag.textContent = `${data.total_latency_ms || data.latency_ms || 0} ms`;
    totalLatencyBadge.textContent = `Pipeline ${data.total_latency_ms || data.latency_ms || 0} ms`;

    if (isTTSActive && data.english_sentence !== lastSpokenSentence) {
      lastSpokenSentence = data.english_sentence;
      playAudioOrSynthesize(data.audio_base64, data.english_sentence);
    }

    const key = `${(data.disambiguated_signs || []).join(" ")}|${data.english_sentence}`;
    if (key !== lastTranscriptKey) {
      lastTranscriptKey = key;
      addTranscriptTurn(data.disambiguated_signs || data.raw_signs, data.english_sentence);
    }
  }

  if (inspectRaw && (data.disambiguated_signs || data.english_sentence)) {
    inspectRaw.textContent = (data.raw_signs || data.disambiguated_signs || []).join(" ") || "--";
    inspectDisambiguated.textContent = (data.disambiguated_signs || []).join(" ") || "--";
    inspectEnglish.textContent = `"${data.english_sentence || "--"}"`;
    inspectSpeech.textContent = data.audio_base64 ? "WAV audio" : (isTTSActive ? "Browser speech" : "Muted");
  }

  const statuses = data.agent_statuses || {};
  updateAgentCard("Vision", statuses.VisionAgent, lastHandsPresent ? "Hands in view" : "Waiting for hands");
  const signText = data.live_hypothesis
    || (data.pending_glosses && data.pending_glosses.join(" "))
    || (data.recognized_signs && data.recognized_signs.length ? data.recognized_signs.map((s) => s.sign).join(" ") : "Waiting");
  updateAgentCard("Sign", statuses.SignRecognitionAgent, signText);
  updateAgentCard("Context", statuses.ContextAgent, data.disambiguated_signs ? data.disambiguated_signs.join(" ") : "Idle");
  updateAgentCard("Language", statuses.LanguageAgent, data.english_sentence || "Idle");
  updateAgentCard("Speech", statuses.SpeechAgent, data.audio_base64 ? "WAV out" : (isTTSActive ? "Browser TTS" : "Muted"));
  updateAgentCard("Memory", statuses.ConversationMemoryAgent, `Turns: ${data.history_turn_count || 0}`);
}

function updateAgentCard(name, statusObj, valueText) {
  const dot = document.getElementById(`dot${name}`);
  const lat = document.getElementById(`lat${name}`);
  const val = document.getElementById(`val${name}`);
  if (!dot || !lat || !val) return;
  if (statusObj) {
    dot.className = `agent-dot ${(statusObj.state || "idle").toLowerCase()}`;
    lat.textContent = `${Math.round(statusObj.latency_ms || 0)}ms`;
  }
  if (valueText) val.textContent = valueText;
}

function playAudioOrSynthesize(audioBase64, text) {
  if (audioBase64) {
    try {
      const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
      audio.play().catch(() => {});
      return;
    } catch (e) { /* fall through */ }
  }
  if ("speechSynthesis" in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

toggleTTS.addEventListener("change", (e) => { isTTSActive = e.target.checked; });

function addTranscriptTurn(signs, translation) {
  const emptyState = transcriptList.querySelector(".empty-state");
  if (emptyState) emptyState.remove();
  const turnEl = document.createElement("div");
  turnEl.className = "transcript-item";
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const signsStr = Array.isArray(signs) ? signs.join(" ") : (signs || "");
  turnEl.innerHTML = `
    <div class="transcript-item-header">
      <span>Turn</span><span>${timeStr}</span>
    </div>
    <div class="transcript-raw">GLOSS: ${signsStr}</div>
    <div class="transcript-english">"${translation}"</div>`;
  transcriptList.prepend(turnEl);
}

btnResetSession.addEventListener("click", async () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "reset" }));
  }
  await fetch("/api/session/reset", { method: "POST" });
  transcriptList.innerHTML = `<div class="empty-state">Session reset.</div>`;
  liveSubtitle.textContent = "Session reset.";
  lastSpokenSentence = "";
  lastTranscriptKey = "";
  renderPhraseChips([]);
  if (liveHypothesis) liveHypothesis.textContent = "—";
  updateAgentCard("Memory", null, "Turns: 0");
});

if (btnCommitPhrase) {
  btnCommitPhrase.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      wsBusy = true;
      ws.send(JSON.stringify({ type: "commit", hands_present: false }));
    }
  });
}

btnExportJson.addEventListener("click", async () => {
  const res = await fetch("/api/transcript/default_session");
  const data = await res.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `asl_transcript_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

function initTestingLab() {
  vocabPallet.innerHTML = "";
  CORE_VOCAB.forEach((word) => {
    const chip = document.createElement("button");
    chip.className = "pallet-word";
    chip.type = "button";
    chip.textContent = word;
    chip.addEventListener("click", () => { activeTokens.push(word); renderActiveTokens(); });
    vocabPallet.appendChild(chip);
  });
  document.querySelectorAll(".preset-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeTokens = btn.getAttribute("data-signs").split(",");
      renderActiveTokens();
      sendSignSequence(activeTokens);
    });
  });
  btnClearTokens.addEventListener("click", () => { activeTokens = []; renderActiveTokens(); });
  btnSendSigns.addEventListener("click", () => {
    if (activeTokens.length > 0) sendSignSequence(activeTokens);
  });
  btnRunBenchmark.addEventListener("click", () => loadBenchmarkTable());
}

function renderActiveTokens() {
  activeTokensContainer.innerHTML = "";
  activeTokens.forEach((token, idx) => {
    const badge = document.createElement("span");
    badge.className = "token-badge";
    badge.innerHTML = `${token} <span class="token-remove">&times;</span>`;
    badge.querySelector(".token-remove").addEventListener("click", (e) => {
      e.stopPropagation();
      activeTokens.splice(idx, 1);
      renderActiveTokens();
    });
    activeTokensContainer.appendChild(badge);
  });
}

function sendSignSequence(signs) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "signs", signs }));
  } else {
    fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signs })
    }).then((r) => r.json()).then(handlePipelineResult);
  }
}

async function loadBenchmarkTable() {
  benchmarkTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:20px;">Running…</td></tr>`;
  try {
    const res = await fetch("/api/benchmark/ambiguity");
    const data = await res.json();
    benchContextAcc.textContent = `${data.context_agent_accuracy_pct}%`;
    benchNoContextAcc.textContent = `${data.no_context_accuracy_pct}%`;
    benchDelta.textContent = `+${data.improvement_pct}%`;
    benchmarkTableBody.innerHTML = "";
    data.results.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.case_id}</td>
        <td><strong>${row.description}</strong></td>
        <td><code>${row.raw_signs.join(" ")}</code></td>
        <td><code>${row.with_context.join(" ")}</code></td>
        <td><code>${row.expected.join(" ")}</code></td>
        <td><span class="${row.context_correct ? "badge-pass" : "badge-fail"}">${row.context_correct ? "RESOLVED" : "FAILED"}</span></td>`;
      benchmarkTableBody.appendChild(tr);
    });
  } catch (e) {
    benchmarkTableBody.innerHTML = `<tr><td colspan="6" style="color:var(--danger);text-align:center;">${e.message}</td></tr>`;
  }
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (backendPill) {
      backendPill.textContent = data.weights_loaded
        ? `Recognizer: ${data.sign_backend}`
        : "Geometry interpreter (no trained BiGRU)";
    }
  } catch (e) { /* ignore */ }
}

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  initTestingLab();
  connectWebSocket();
  initMediaPipe();
  startCameraStream();
  loadHealth();
});
