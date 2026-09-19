/**
 * SignVoice client: auth gate, landing page, camera, MediaPipe Hands, WebSocket pipeline.
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

// ── Auth state ────────────────────────────────────────────────────────────────

const AUTH_TOKEN_KEY = "sv_auth_token";
const AUTH_USER_KEY  = "sv_auth_user";

function saveAuth(token, user) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

function getAuthUser() {
  try { return JSON.parse(localStorage.getItem(AUTH_USER_KEY) || "null"); } catch { return null; }
}

function clearAuth() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
}

async function verifyToken(token) {
  try {
    const res = await fetch("/api/auth/verify", {
      headers: { "Authorization": `Bearer ${token}` }
    });
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

async function loginRequest(email, password, name) {
  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name: name || undefined })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(err.detail || "Login failed");
  }
  return res.json();
}

// ── View management ───────────────────────────────────────────────────────────

function showLanding() {
  document.getElementById("viewLanding").style.display = "";
  document.getElementById("appShell").style.display = "none";
  document.getElementById("authModal").style.display = "none";
}

function showApp(user) {
  document.getElementById("viewLanding").style.display = "none";
  document.getElementById("authModal").style.display = "none";
  document.getElementById("appShell").style.display = "";
  applyUserBadge(user);
  applyRoleGating(user.role);
}

function openAuthModal(mode) {
  const modal = document.getElementById("authModal");
  modal.style.display = "flex";
  if (mode === "signup") {
    document.getElementById("formLogin").style.display = "none";
    document.getElementById("formSignup").style.display = "";
  } else {
    document.getElementById("formLogin").style.display = "";
    document.getElementById("formSignup").style.display = "none";
  }
}

function closeAuthModal() {
  document.getElementById("authModal").style.display = "none";
}

function applyUserBadge(user) {
  const avatarEl = document.getElementById("userAvatar");
  const nameEl   = document.getElementById("userNameDisplay");
  const roleEl   = document.getElementById("userRoleChip");
  if (!avatarEl) return;
  avatarEl.textContent = (user.name || user.email || "?")[0].toUpperCase();
  nameEl.textContent   = user.name || user.email;
  roleEl.textContent   = user.role === "admin" ? "Admin" : "User";
  if (user.role === "admin") roleEl.classList.add("admin");
  else roleEl.classList.remove("admin");
}

function applyRoleGating(role) {
  const testingTab = document.getElementById("tabBtnTesting");
  if (!testingTab) return;
  if (role === "admin") {
    testingTab.style.display = "";
  } else {
    testingTab.style.display = "none";
    // Force dashboard view if someone tries to access testing
    const viewTesting = document.getElementById("viewTesting");
    const viewDashboard = document.getElementById("viewDashboard");
    if (viewTesting) viewTesting.style.display = "none";
    if (viewDashboard) viewDashboard.style.display = "";
  }
}

// ── Interpreter state ─────────────────────────────────────────────────────────

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

// Lazily resolved after app shell is shown
let wsStatus, videoElement, canvasElement, canvasCtx, videoPlaceholder,
    fpsDisplay, bufferDisplay, handsDisplay, liveHypothesis, confBar,
    holdBar, holdLabel, sensitivitySelect, phraseChips, backendPill,
    liveSubtitle, providerTag, latencyTag, totalLatencyBadge,
    toggleWebcam, camSwitchLabel, toggleTTS, activeTokensContainer,
    vocabPallet, btnClearTokens, btnSendSigns, transcriptList,
    btnExportJson, btnResetSession, btnCommitPhrase,
    tabBtnDashboard, tabBtnTesting, viewDashboard, viewTesting,
    inspectRaw, inspectDisambiguated, inspectEnglish, inspectSpeech,
    btnRunBenchmark, benchContextAcc, benchNoContextAcc, benchDelta,
    benchmarkTableBody, currentUserRole;

function resolveAppElements() {
  wsStatus            = document.getElementById("wsStatus");
  videoElement        = document.getElementById("webcamVideo");
  canvasElement       = document.getElementById("landmarkCanvas");
  canvasCtx           = canvasElement.getContext("2d");
  videoPlaceholder    = document.getElementById("videoPlaceholder");
  fpsDisplay          = document.getElementById("fpsDisplay");
  bufferDisplay       = document.getElementById("bufferDisplay");
  handsDisplay        = document.getElementById("handsDisplay");
  liveHypothesis      = document.getElementById("liveHypothesis");
  confBar             = document.getElementById("confBar");
  holdBar             = document.getElementById("holdBar");
  holdLabel           = document.getElementById("holdLabel");
  sensitivitySelect   = document.getElementById("sensitivitySelect");
  phraseChips         = document.getElementById("phraseChips");
  backendPill         = document.getElementById("backendPill");
  liveSubtitle        = document.getElementById("liveSubtitle");
  providerTag         = document.getElementById("providerTag");
  latencyTag          = document.getElementById("latencyTag");
  totalLatencyBadge   = document.getElementById("totalLatencyBadge");
  toggleWebcam        = document.getElementById("toggleWebcam");
  camSwitchLabel      = document.getElementById("camSwitchLabel");
  toggleTTS           = document.getElementById("toggleTTS");
  activeTokensContainer = document.getElementById("activeTokens");
  vocabPallet         = document.getElementById("vocabPallet");
  btnClearTokens      = document.getElementById("btnClearTokens");
  btnSendSigns        = document.getElementById("btnSendSigns");
  transcriptList      = document.getElementById("transcriptList");
  btnExportJson       = document.getElementById("btnExportJson");
  btnResetSession     = document.getElementById("btnResetSession");
  btnCommitPhrase     = document.getElementById("btnCommitPhrase");
  tabBtnDashboard     = document.getElementById("tabBtnDashboard");
  tabBtnTesting       = document.getElementById("tabBtnTesting");
  viewDashboard       = document.getElementById("viewDashboard");
  viewTesting         = document.getElementById("viewTesting");
  inspectRaw          = document.getElementById("inspectRaw");
  inspectDisambiguated= document.getElementById("inspectDisambiguated");
  inspectEnglish      = document.getElementById("inspectEnglish");
  inspectSpeech       = document.getElementById("inspectSpeech");
  btnRunBenchmark     = document.getElementById("btnRunBenchmark");
  benchContextAcc     = document.getElementById("benchContextAcc");
  benchNoContextAcc   = document.getElementById("benchNoContextAcc");
  benchDelta          = document.getElementById("benchDelta");
  benchmarkTableBody  = document.getElementById("benchmarkTableBody");
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

function setupTabs() {
  tabBtnDashboard.addEventListener("click", () => {
    tabBtnDashboard.classList.add("active");
    tabBtnTesting.classList.remove("active");
    viewDashboard.style.display = "";
    viewTesting.style.display = "none";
  });

  tabBtnTesting.addEventListener("click", () => {
    // Double-check role before allowing access
    if (currentUserRole !== "admin") return;
    tabBtnTesting.classList.add("active");
    tabBtnDashboard.classList.remove("active");
    viewDashboard.style.display = "none";
    viewTesting.style.display = "grid";
    loadBenchmarkTable();
  });
}

// ── Camera ────────────────────────────────────────────────────────────────────

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
  if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  videoElement.srcObject = null;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  videoPlaceholder.classList.add("visible");
  camSwitchLabel.textContent = "Camera off";
  fpsDisplay.textContent = "OFF";
  bufferDisplay.textContent = "Standby";
}

// ── MediaPipe ─────────────────────────────────────────────────────────────────

function initMediaPipe() {
  if (typeof Hands === "undefined") { console.warn("MediaPipe Hands library not loaded."); return; }
  handsDetector = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
  handsDetector.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.55, minTrackingConfidence: 0.5 });
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
  if (now - lastFpsUpdate > 300) { fpsDisplay.textContent = `${frameTimes.length} FPS`; lastFpsUpdate = now; }
  if (handsDetector && !isProcessingFrame) {
    isProcessingFrame = true;
    handsDetector.send({ image: videoElement }).catch((e) => console.warn("Hands detection skipped frame:", e)).finally(() => { isProcessingFrame = false; });
  }
  if (isWebcamActive) animationFrameId = requestAnimationFrame(runVideoDetectionLoop);
}

function packLandmarks(results) {
  const lh = new Array(63).fill(0.0);
  const rh = new Array(63).fill(0.0);
  if (!results.multiHandedness || !results.multiHandLandmarks) return lh.concat(rh);
  results.multiHandedness.forEach((handMeta, idx) => {
    const isRight = handMeta.label === "Left";
    const landmarks = results.multiHandLandmarks[idx];
    const target = isRight ? rh : lh;
    landmarks.forEach((pt, i) => {
      if (i < 21) { target[i * 3] = pt.x; target[i * 3 + 1] = pt.y; target[i * 3 + 2] = pt.z; }
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
  ws.send(JSON.stringify({ type: "landmarks", landmarks: [featureVector], hands_present: handsPresent, sensitivity: sensitivitySelect ? sensitivitySelect.value : "steady" }));
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
        const pa = landmarks[a]; const pb = landmarks[b];
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

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
  ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    wsBusy = false;
    wsStatus.querySelector(".status-dot").classList.add("connected");
    wsStatus.querySelector(".status-label").textContent = "Live";
    if (sensitivitySelect) ws.send(JSON.stringify({ type: "sensitivity", sensitivity: sensitivitySelect.value }));
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
    try { handlePipelineResult(JSON.parse(event.data)); } catch (e) { console.error("Error parsing WebSocket message:", e); }
  };
}

// ── Pipeline result handler ───────────────────────────────────────────────────

function handlePipelineResult(data) {
  if (!data) return;
  if (backendPill && data.sign_backend) {
    const label = data.sign_backend === "geometric" || data.sign_backend === "geometric_fallback"
      ? "Geometry interpreter (no trained BiGRU)"
      : `Recognizer: ${data.sign_backend}`;
    backendPill.textContent = label;
  }
  if (bufferDisplay && data.buffer_len != null) bufferDisplay.textContent = `Buffer ${data.buffer_len}/30`;
  if (liveHypothesis) liveHypothesis.textContent = data.live_hypothesis || "—";
  if (confBar) { const pct = Math.round((data.live_confidence || 0) * 100); confBar.style.width = `${pct}%`; }
  if (holdBar) { const holdPct = Math.round((data.hold_progress || 0) * 100); holdBar.style.width = `${holdPct}%`; }
  if (holdLabel) {
    if (data.hold_progress >= 0.95) { holdLabel.textContent = "Locked ✓"; holdLabel.style.color = "var(--success)"; }
    else if (data.hold_progress > 0) { holdLabel.textContent = "Locking in…"; holdLabel.style.color = "var(--accent)"; }
    else { holdLabel.textContent = "Hold to commit"; holdLabel.style.color = "var(--accent-2)"; }
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
    if (key !== lastTranscriptKey) { lastTranscriptKey = key; addTranscriptTurn(data.disambiguated_signs || data.raw_signs, data.english_sentence); }
  }
  if (inspectRaw && (data.disambiguated_signs || data.english_sentence)) {
    inspectRaw.textContent = (data.raw_signs || data.disambiguated_signs || []).join(" ") || "--";
    inspectDisambiguated.textContent = (data.disambiguated_signs || []).join(" ") || "--";
    inspectEnglish.textContent = `"${data.english_sentence || "--"}"`;
    inspectSpeech.textContent = data.audio_base64 ? "WAV audio" : (isTTSActive ? "Browser speech" : "Muted");
  }
  const statuses = data.agent_statuses || {};
  updateAgentCard("Vision", statuses.VisionAgent, lastHandsPresent ? "Hands in view" : "Waiting for hands");
  const signText = data.live_hypothesis || (data.pending_glosses && data.pending_glosses.join(" ")) || (data.recognized_signs && data.recognized_signs.length ? data.recognized_signs.map((s) => s.sign).join(" ") : "Waiting");
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
  if (statusObj) { dot.className = `agent-dot ${(statusObj.state || "idle").toLowerCase()}`; lat.textContent = `${Math.round(statusObj.latency_ms || 0)}ms`; }
  if (valueText) val.textContent = valueText;
}

function renderPhraseChips(glosses) {
  if (!phraseChips) return;
  if (!glosses || !glosses.length) { phraseChips.innerHTML = `<span class="muted">Signs you hold will appear here</span>`; return; }
  phraseChips.innerHTML = glosses.map((g) => `<span class="chip">${g}</span>`).join("");
}

function playAudioOrSynthesize(audioBase64, text) {
  if (audioBase64) {
    try { const audio = new Audio(`data:audio/wav;base64,${audioBase64}`); audio.play().catch(() => {}); return; } catch (e) { /* fall through */ }
  }
  if ("speechSynthesis" in window) { const utterance = new SpeechSynthesisUtterance(text); utterance.rate = 1.0; window.speechSynthesis.speak(utterance); }
}

function addTranscriptTurn(signs, translation) {
  const emptyState = transcriptList.querySelector(".empty-state");
  if (emptyState) emptyState.remove();
  const turnEl = document.createElement("div");
  turnEl.className = "transcript-item";
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const signsStr = Array.isArray(signs) ? signs.join(" ") : (signs || "");
  turnEl.innerHTML = `
    <div class="transcript-item-header"><span>Turn</span><span>${timeStr}</span></div>
    <div class="transcript-raw">GLOSS: ${signsStr}</div>
    <div class="transcript-english">"${translation}"</div>`;
  transcriptList.prepend(turnEl);
}

// ── Testing lab ───────────────────────────────────────────────────────────────

function initTestingLab() {
  if (!vocabPallet) return;
  vocabPallet.innerHTML = "";
  CORE_VOCAB.forEach((word) => {
    const chip = document.createElement("button");
    chip.className = "pallet-word"; chip.type = "button"; chip.textContent = word;
    chip.addEventListener("click", () => { activeTokens.push(word); renderActiveTokens(); });
    vocabPallet.appendChild(chip);
  });
  document.querySelectorAll(".preset-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (currentUserRole !== "admin") return;
      activeTokens = btn.getAttribute("data-signs").split(",");
      renderActiveTokens();
      sendSignSequence(activeTokens);
    });
  });
  if (btnClearTokens) btnClearTokens.addEventListener("click", () => { activeTokens = []; renderActiveTokens(); });
  if (btnSendSigns) btnSendSigns.addEventListener("click", () => { if (activeTokens.length > 0) sendSignSequence(activeTokens); });
  if (btnRunBenchmark) btnRunBenchmark.addEventListener("click", () => loadBenchmarkTable());
}

function renderActiveTokens() {
  if (!activeTokensContainer) return;
  activeTokensContainer.innerHTML = "";
  activeTokens.forEach((token, idx) => {
    const badge = document.createElement("span");
    badge.className = "token-badge";
    badge.innerHTML = `${token} <span class="token-remove">&times;</span>`;
    badge.querySelector(".token-remove").addEventListener("click", (e) => { e.stopPropagation(); activeTokens.splice(idx, 1); renderActiveTokens(); });
    activeTokensContainer.appendChild(badge);
  });
}

function sendSignSequence(signs) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "signs", signs }));
  } else {
    fetch("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ signs }) })
      .then((r) => r.json()).then(handlePipelineResult);
  }
}

async function loadBenchmarkTable() {
  if (!benchmarkTableBody) return;
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
    if (backendPill) backendPill.textContent = data.weights_loaded ? `Recognizer: ${data.sign_backend}` : "Geometry interpreter (no trained BiGRU)";
  } catch (e) { /* ignore */ }
}

// ── Bind app-level events (after elements are resolved) ───────────────────────

function bindAppEvents() {
  if (toggleWebcam) {
    toggleWebcam.addEventListener("change", (e) => {
      if (e.target.checked) startCameraStream();
      else stopCameraStream();
    });
  }
  if (sensitivitySelect) {
    const saved = localStorage.getItem("asl_sensitivity") || "steady";
    sensitivitySelect.value = saved;
    sensitivitySelect.addEventListener("change", () => {
      localStorage.setItem("asl_sensitivity", sensitivitySelect.value);
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "sensitivity", sensitivity: sensitivitySelect.value }));
    });
  }
  if (toggleTTS) toggleTTS.addEventListener("change", (e) => { isTTSActive = e.target.checked; });
  if (btnResetSession) {
    btnResetSession.addEventListener("click", async () => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
      await fetch("/api/session/reset", { method: "POST" });
      transcriptList.innerHTML = `<div class="empty-state">Session reset.</div>`;
      liveSubtitle.textContent = "Session reset.";
      lastSpokenSentence = ""; lastTranscriptKey = "";
      renderPhraseChips([]);
      if (liveHypothesis) liveHypothesis.textContent = "—";
      updateAgentCard("Memory", null, "Turns: 0");
    });
  }
  if (btnCommitPhrase) {
    btnCommitPhrase.addEventListener("click", () => {
      if (ws && ws.readyState === WebSocket.OPEN) { wsBusy = true; ws.send(JSON.stringify({ type: "commit", hands_present: false })); }
    });
  }
  if (btnExportJson) {
    btnExportJson.addEventListener("click", async () => {
      const res = await fetch("/api/transcript/default_session");
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = `asl_transcript_${Date.now()}.json`; a.click();
      URL.revokeObjectURL(url);
    });
  }
  if (btnLogout) {
    document.getElementById("btnLogout").addEventListener("click", () => {
      clearAuth();
      // Stop camera and WebSocket cleanly
      if (isWebcamActive) stopCameraStream();
      if (ws) { ws.onclose = null; ws.close(); ws = null; }
      showLanding();
    });
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function bootApp(user) {
  currentUserRole = user.role;
  showApp(user);
  resolveAppElements();
  setupTabs();
  bindAppEvents();
  initTestingLab();
  connectWebSocket();
  initMediaPipe();
  startCameraStream();
  loadHealth();
}

window.addEventListener("DOMContentLoaded", async () => {
  // ── Landing page CTA wiring ──
  document.getElementById("landingLoginBtn").addEventListener("click",  () => openAuthModal("login"));
  document.getElementById("landingSignupBtn").addEventListener("click", () => openAuthModal("signup"));
  document.getElementById("heroGetStartedBtn").addEventListener("click",() => openAuthModal("signup"));
  document.getElementById("heroLearnBtn").addEventListener("click", () => {
    document.getElementById("features").scrollIntoView({ behavior: "smooth" });
  });
  document.getElementById("ctaSignupBtn").addEventListener("click", () => openAuthModal("signup"));

  // ── Modal wiring ──
  document.getElementById("modalCloseBtn").addEventListener("click", closeAuthModal);
  document.getElementById("authModal").addEventListener("click", (e) => { if (e.target === document.getElementById("authModal")) closeAuthModal(); });
  document.getElementById("switchToSignup").addEventListener("click", () => {
    document.getElementById("formLogin").style.display = "none";
    document.getElementById("formSignup").style.display = "";
  });
  document.getElementById("switchToLogin").addEventListener("click", () => {
    document.getElementById("formSignup").style.display = "none";
    document.getElementById("formLogin").style.display = "";
  });

  // ── Login form ──
  document.getElementById("btnLogin").addEventListener("click", async () => {
    const email    = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;
    const errEl    = document.getElementById("loginError");
    errEl.style.display = "none";
    if (!email || !password) { errEl.textContent = "Please fill in all fields."; errEl.style.display = ""; return; }
    const btn = document.getElementById("btnLogin");
    btn.disabled = true; btn.textContent = "Signing in…";
    try {
      const data = await loginRequest(email, password, null);
      saveAuth(data.token, { email: data.email, name: data.name, role: data.role });
      closeAuthModal();
      await bootApp({ email: data.email, name: data.name, role: data.role });
    } catch (e) {
      errEl.textContent = e.message; errEl.style.display = "";
    } finally {
      btn.disabled = false; btn.textContent = "Log In";
    }
  });

  // Allow Enter key on login form
  document.getElementById("loginPassword").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("btnLogin").click();
  });

  // ── Signup form ──
  document.getElementById("btnSignup").addEventListener("click", async () => {
    const name     = document.getElementById("signupName").value.trim();
    const email    = document.getElementById("signupEmail").value.trim();
    const password = document.getElementById("signupPassword").value;
    const errEl    = document.getElementById("signupError");
    errEl.style.display = "none";
    if (!name || !email || !password) { errEl.textContent = "Please fill in all fields."; errEl.style.display = ""; return; }
    if (password.length < 6) { errEl.textContent = "Password must be at least 6 characters."; errEl.style.display = ""; return; }
    const btn = document.getElementById("btnSignup");
    btn.disabled = true; btn.textContent = "Creating account…";
    try {
      const data = await loginRequest(email, password, name);
      saveAuth(data.token, { email: data.email, name: data.name, role: data.role });
      closeAuthModal();
      await bootApp({ email: data.email, name: data.name, role: data.role });
    } catch (e) {
      errEl.textContent = e.message; errEl.style.display = "";
    } finally {
      btn.disabled = false; btn.textContent = "Create Account";
    }
  });

  // ── Check for existing session ──
  const token = getAuthToken();
  const cachedUser = getAuthUser();
  if (token && cachedUser) {
    // Verify server-side that the token is still valid
    const verified = await verifyToken(token);
    if (verified && verified.valid) {
      await bootApp({ email: verified.email, name: verified.name, role: verified.role });
      return;
    } else {
      clearAuth();
    }
  }
  // No valid session — show landing
  showLanding();
});
