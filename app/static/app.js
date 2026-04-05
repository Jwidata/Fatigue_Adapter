const caseSelect = document.getElementById("case-select");
const sliceInput = document.getElementById("slice-input");
const loadSliceBtn = document.getElementById("load-slice");
const sliceImage = document.getElementById("slice-image");
const roiCanvas = document.getElementById("roi-canvas");
const gazeCanvas = document.getElementById("gaze-canvas");
const toggleRoi = document.getElementById("toggle-roi");
const toggleGaze = document.getElementById("toggle-gaze");
const togglePredict = document.getElementById("toggle-predict");
const toggleAdapt = document.getElementById("toggle-adapt");
const toggleDebug = document.getElementById("toggle-debug");
const sourceSelect = document.getElementById("source-select");
const modeSelect = document.getElementById("mode-select");
const policySelect = document.getElementById("policy-select");
const messageMinRisk = document.getElementById("message-min-risk");
const interventionMinRisk = document.getElementById("intervention-min-risk");
const cooldownMs = document.getElementById("cooldown-ms");
const messageHoldMs = document.getElementById("message-hold-ms");
const silentLowRisk = document.getElementById("silent-low-risk");
const resetButton = document.getElementById("reset-button");
const statePill = document.getElementById("state-pill");
const sourcePill = document.getElementById("source-pill");
const sidePanel = document.getElementById("side-panel");
const adaptationCard = document.getElementById("adaptation-card");
const statusMain = document.getElementById("status-main");
const statusAttention = document.getElementById("status-attention");
const statusCoverage = document.getElementById("status-coverage");
const statusRisk = document.getElementById("status-risk");
const statusAction = document.getElementById("status-action");
const statusOutcome = document.getElementById("status-outcome");
const statusSuggest = document.getElementById("status-suggest");
const statusGaze = document.getElementById("status-gaze");
const statusPred = document.getElementById("status-pred");
const statusRoi = document.getElementById("status-roi");
const statusThreshold = document.getElementById("status-threshold");
const statusMode = document.getElementById("status-mode");
const predMode = document.getElementById("pred-mode");
const predActive = document.getElementById("pred-active");
const predExplain = document.getElementById("pred-explain");
const predStatus = document.getElementById("pred-status");
const predConfidence = document.getElementById("pred-confidence");
const predAccuracy = document.getElementById("pred-accuracy");
const imageStage = document.getElementById("image-stage");
const imageOverlay = document.getElementById("image-overlay");
const debugPanel = document.getElementById("debug-panel");

const metrics = {
  dwell: document.getElementById("metric-dwell"),
  hits: document.getElementById("metric-hits"),
  delay: document.getElementById("metric-delay"),
  coverage: document.getElementById("metric-coverage"),
  dispersion: document.getElementById("metric-dispersion"),
  scan: document.getElementById("metric-scan"),
  predMean: document.getElementById("metric-pred-mean"),
  predMedian: document.getElementById("metric-pred-median"),
  predWithin: document.getElementById("metric-pred-within"),
};

const adaptation = {
  actions: document.getElementById("adaptation-actions"),
  message: document.getElementById("adaptation-message"),
};

let currentCase = null;
let currentSlice = 0;
let currentRoi = [];
let imageDims = { width: 0, height: 0 };
let lastMouseSent = 0;
let latestPrediction = null;
let latestGaze = null;
const gazeTrail = [];
const maxTrail = 12;
let lastMessageUpdate = 0;
let lastMessageText = "";
let policyInfo = null;
let currentRiskLevel = "low";

async function fetchCases() {
  const response = await fetch("/api/cases");
  const data = await response.json();
  caseSelect.innerHTML = "";
  data.cases.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.case_id;
    option.textContent = entry.case_id;
    caseSelect.appendChild(option);
  });
  if (data.cases.length > 0) {
    currentCase = data.cases[0].case_id;
    caseSelect.value = currentCase;
  }
}

async function loadSlice() {
  if (!currentCase) return;
  currentSlice = parseInt(sliceInput.value, 10) || 0;
  const imgUrl = `/api/cases/${currentCase}/slices/${currentSlice}`;
  sliceImage.src = imgUrl;
  await updateMode();
  await fetchRoi();
}

async function fetchRoi() {
  if (!currentCase) return;
  const response = await fetch(`/api/roi/${currentCase}/${currentSlice}`);
  const data = await response.json();
  currentRoi = data.rois || [];
  imageDims = { width: data.image_width, height: data.image_height };
  drawRoi();
}

function resizeCanvas() {
  const rect = sliceImage.getBoundingClientRect();
  roiCanvas.width = rect.width;
  roiCanvas.height = rect.height;
  gazeCanvas.width = rect.width;
  gazeCanvas.height = rect.height;
}

function drawRoi() {
  resizeCanvas();
  const ctx = roiCanvas.getContext("2d");
  ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  if (!toggleRoi.checked) return;
  if (!currentRoi.length) {
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(16, 16, 240, 26);
    ctx.fillStyle = "#fff";
    ctx.font = "13px 'Palatino Linotype', serif";
    ctx.fillText("No ROI available for this slice", 24, 34);
    return;
  }
  ctx.strokeStyle = "rgba(255, 214, 10, 0.95)";
  ctx.lineWidth = 4;
  ctx.fillStyle = "rgba(255, 214, 10, 0.25)";
  currentRoi.forEach((roi) => {
    if (roi.type === "bbox" && roi.bbox) {
      const x = roi.bbox.x * roiCanvas.width;
      const y = roi.bbox.y * roiCanvas.height;
      const w = roi.bbox.w * roiCanvas.width;
      const h = roi.bbox.h * roiCanvas.height;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
      ctx.fillRect(x, Math.max(0, y - 22), 150, 20);
      ctx.fillStyle = "#fff";
      ctx.font = "12px 'Palatino Linotype', serif";
      ctx.fillText("Critical region", x + 6, Math.max(14, y - 8));
      ctx.fillStyle = "rgba(255, 214, 10, 0.2)";
    }
  });
}

function drawGaze(point, prediction) {
  resizeCanvas();
  const ctx = gazeCanvas.getContext("2d");
  ctx.clearRect(0, 0, gazeCanvas.width, gazeCanvas.height);
  if (!toggleGaze.checked || !point || imageDims.width === 0 || imageDims.height === 0) return;
  const x = (point.x / imageDims.width) * gazeCanvas.width;
  const y = (point.y / imageDims.height) * gazeCanvas.height;

  if (toggleDebug.checked) {
    gazeTrail.push({ x, y, t: Date.now() });
    if (gazeTrail.length > maxTrail) gazeTrail.shift();
    ctx.strokeStyle = "rgba(47, 128, 255, 0.25)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    gazeTrail.forEach((p, idx) => {
      if (idx === 0) {
        ctx.moveTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }
    });
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(47, 128, 255, 0.95)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 2;
  ctx.stroke();

  if (prediction) {
    const px = (prediction.x / imageDims.width) * gazeCanvas.width;
    const py = (prediction.y / imageDims.height) * gazeCanvas.height;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(px, py);
    ctx.strokeStyle = "rgba(0, 209, 255, 0.6)";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0, 209, 255, 0.9)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function isGazeInRoi(point) {
  if (!point || !currentRoi.length) return false;
  return currentRoi.some((roi) => {
    if (roi.type !== "bbox" || !roi.bbox) return false;
    const x0 = roi.bbox.x * imageDims.width;
    const y0 = roi.bbox.y * imageDims.height;
    const x1 = x0 + roi.bbox.w * imageDims.width;
    const y1 = y0 + roi.bbox.h * imageDims.height;
    return point.x >= x0 && point.x <= x1 && point.y >= y0 && point.y <= y1;
  });
}

function updateMetrics(data) {
  metrics.dwell.textContent = `Dwell: ${data.metrics.dwell_time_ms.toFixed(0)} ms`;
  metrics.hits.textContent = `ROI hits: ${data.metrics.roi_hits}`;
  metrics.delay.textContent = `First fixation: ${data.metrics.first_fixation_delay_ms ?? "--"} ms`;
  metrics.coverage.textContent = `ROI coverage: ${data.metrics.roi_coverage_pct.toFixed(1)}%`;
  metrics.dispersion.textContent = `Dispersion: ${data.metrics.dispersion.toFixed(3)}`;
  metrics.scan.textContent = `Scan coverage: ${data.metrics.scan_coverage_pct.toFixed(1)}%`;
  statePill.textContent = `State: ${data.state}`;
  sourcePill.textContent = `Source: ${sourceSelect.value}`;

  const emoji = data.state === "at_risk" ? "⚠️" : data.state === "drifting_attention" ? "🧭" : data.state === "low_roi_engagement" ? "👀" : "✅";
  statusMain.textContent = `State: ${data.state} ${emoji}`;

  const inRoi = isGazeInRoi(data.latest_gaze);
  if (inRoi) {
    statusAttention.innerHTML = `<span class="indicator-good">Looking at important region</span>`;
  } else {
    statusAttention.innerHTML = `<span class="indicator-bad">Missing important region</span>`;
  }
  const coverageLabel = data.metrics.roi_coverage_pct < 25 ? "Low" : data.metrics.roi_coverage_pct < 60 ? "Medium" : "High";
  statusCoverage.textContent = `ROI coverage: ${coverageLabel}`;
  if (data.latest_gaze) {
    statusGaze.textContent = `Current gaze: (${data.latest_gaze.x.toFixed(1)}, ${data.latest_gaze.y.toFixed(1)})`;
  } else {
    statusGaze.textContent = "Current gaze: --";
  }
  if (data.prediction && data.prediction.predicted) {
    statusPred.textContent = `Predicted next look: (${data.prediction.predicted.x.toFixed(1)}, ${data.prediction.predicted.y.toFixed(1)})`;
  } else {
    statusPred.textContent = "Predicted next look: --";
  }

  if (data.risk) {
    statusRisk.textContent = `Risk: ${data.risk.risk_level.toUpperCase()}`;
    currentRiskLevel = data.risk.risk_level;
  } else {
    statusRisk.textContent = "Risk: --";
    currentRiskLevel = "low";
  }

  if (data.latest_gaze) {
    statusGaze.textContent = `Current gaze: (${data.latest_gaze.x.toFixed(1)}, ${data.latest_gaze.y.toFixed(1)})`;
  } else {
    statusGaze.textContent = "Current gaze: --";
  }
  if (data.prediction && data.prediction.predicted) {
    statusPred.textContent = `Predicted next look: (${data.prediction.predicted.x.toFixed(1)}, ${data.prediction.predicted.y.toFixed(1)})`;
  } else {
    statusPred.textContent = "Predicted next look: --";
  }
  const roiVisible = currentRoi.length > 0 && toggleRoi.checked;
  statusRoi.textContent = `Critical region: ${roiVisible ? "visible" : "hidden"}`;
}

function updateAdaptation(adapt) {
  const statusText = resolveAdaptationStatus(adapt.command.actions);
  adaptation.actions.textContent = statusText;
  adaptation.message.textContent = resolveAdaptationDetail(adapt.command.actions);

  if (!toggleAdapt.checked) {
    sidePanel.classList.remove("dimmed");
    imageStage.classList.remove("dimmed", "glow");
    imageOverlay.classList.remove("active");
    return;
  }
  if (adapt.command.actions.includes("dim_non_roi")) {
    sidePanel.classList.add("dimmed");
    imageStage.classList.add("dimmed");
  } else {
    sidePanel.classList.remove("dimmed");
    if (!adapt.command.actions.includes("show_focus_prompt")) {
      imageStage.classList.remove("dimmed");
    }
  }

  if (adapt.command.actions.includes("highlight_roi") || adapt.command.actions.includes("strong_highlight_roi")) {
    imageStage.classList.add("glow");
  } else {
    imageStage.classList.remove("glow");
  }
  if (adapt.command.actions.length > 0) {
    imageOverlay.classList.add("active");
    imageOverlay.textContent = adapt.command.message || "Predicted attention drift detected — review highlighted region";
  } else {
    imageOverlay.classList.remove("active");
  }
}

async function updateMode() {
  if (!currentCase) return;
  policyInfo = null;
  await fetch("/api/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gaze_mode: modeSelect.value,
      gaze_source: sourceSelect.value,
      case_id: currentCase,
      slice_id: currentSlice,
      predictive_enabled: togglePredict.checked,
      adaptive_enabled: toggleAdapt.checked,
      policy_mode: policySelect.value,
      message_min_risk: messageMinRisk.value,
      intervention_min_risk: interventionMinRisk.value,
      cooldown_ms: parseInt(cooldownMs.value, 10) || 0,
      message_hold_ms: parseInt(messageHoldMs.value, 10) || 0,
      silent_low_risk: silentLowRisk.checked,
    }),
  });
}

async function pollState() {
  const response = await fetch("/api/state");
  const data = await response.json();
  latestPrediction = data.prediction ? data.prediction.predicted : null;
  latestGaze = data.latest_gaze;
  updateMetrics(data);
  drawGaze(data.latest_gaze, latestPrediction);
  const adaptResponse = await fetch("/api/adaptation");
  const adapt = await adaptResponse.json();
  updateAdaptation(adapt);
  statusAction.textContent = `Action: ${formatActions(adapt.command.actions)}`;

  await updatePolicyInfo(data, adapt);
  await updatePredictionTransparency(data);

  const outcomeResponse = await fetch("/api/adaptation/outcome");
  const outcome = await outcomeResponse.json();
  if (outcome && outcome.action) {
    if (outcome.success === true) {
      statusOutcome.textContent = `Outcome: Success in ${Math.round(outcome.time_to_roi_ms)} ms`;
    } else if (outcome.success === false) {
      statusOutcome.textContent = "Outcome: No fixation yet";
    } else {
      statusOutcome.textContent = "Outcome: Tracking";
    }
  } else {
    statusOutcome.textContent = "Outcome: --";
  }
}

async function updatePolicyInfo(stateData, adapt) {
  if (!policyInfo) {
    const response = await fetch("/api/policy/info");
    policyInfo = await response.json();
  }
  const riskLevel = stateData.risk ? stateData.risk.risk_level : "low";
  const thresholdReached = isRiskAtLeast(riskLevel, policyInfo.intervention_min_risk);
  statusThreshold.textContent = `Adaptation threshold reached: ${thresholdReached ? "yes" : "no"}`;

  let modeLabel = "Monitoring";
  if (adapt.command.actions.length > 0) {
    modeLabel = "Intervention";
  } else if (thresholdReached) {
    modeLabel = "Armed";
  }
  statusMode.textContent = `Current mode: ${modeLabel}`;

  updateSuggestionText(riskLevel, modeLabel);
}

async function updatePredictionTransparency(stateData) {
  const response = await fetch("/api/prediction/info");
  const info = await response.json();
  predMode.textContent = `Predictor mode: ${info.predictor_mode}`;
  predActive.textContent = `Active predictor: ${info.active_predictor}`;
  predExplain.textContent = `Explanation: ${info.explanation}`;
  predStatus.textContent = `Model status: ${info.model_status}`;
  predConfidence.textContent = info.confidence !== null && info.confidence !== undefined
    ? `Confidence: ${info.confidence.toFixed(2)}`
    : "Confidence: not available";
  predAccuracy.textContent = `Accuracy: ${info.accuracy_note}`;

  if (toggleDebug.checked && info.eval_stats) {
    if (info.eval_stats.count >= 10) {
      metrics.predMean.textContent = `Prediction error mean: ${info.eval_stats.mean_error_px.toFixed(1)} px`;
      metrics.predMedian.textContent = `Prediction error median: ${info.eval_stats.median_error_px.toFixed(1)} px`;
      metrics.predWithin.textContent = `Prediction within 25/50 px: ${info.eval_stats.within_25_px_pct.toFixed(1)}% / ${info.eval_stats.within_50_px_pct.toFixed(1)}%`;
    } else {
      metrics.predMean.textContent = "Prediction error mean: Not enough samples yet";
      metrics.predMedian.textContent = "Prediction error median: Not enough samples yet";
      metrics.predWithin.textContent = "Prediction within 25/50 px: Not enough samples yet";
    }
  } else if (toggleDebug.checked) {
    metrics.predMean.textContent = "Prediction error mean: Not enough samples yet";
    metrics.predMedian.textContent = "Prediction error median: Not enough samples yet";
    metrics.predWithin.textContent = "Prediction within 25/50 px: Not enough samples yet";
  }
}

function updateSuggestionText(riskLevel, modeLabel) {
  const now = Date.now();
  const messageHold = policyInfo ? policyInfo.message_hold_ms : 1200;
  const minRisk = policyInfo ? policyInfo.message_min_risk : "medium";
  const shouldMessage = isRiskAtLeast(riskLevel, minRisk) || modeLabel === "Intervention";
  const silentLowRisk = policyInfo ? policyInfo.silent_low_risk : false;

  let text = "Status: monitoring";
  if (shouldMessage) {
    text = modeLabel === "Intervention"
      ? "Suggestion: Review highlighted region"
      : `Suggestion: Risk ${riskLevel.toUpperCase()} — stay attentive`;
  } else if (silentLowRisk) {
    text = "";
  }

  if (text !== lastMessageText || now - lastMessageUpdate > messageHold) {
    statusSuggest.textContent = text;
    lastMessageText = text;
    lastMessageUpdate = now;
  }
}

function formatActions(actions) {
  if (!actions || !actions.length) return "no intervention";
  const map = {
    highlight_roi: "highlight",
    strong_highlight_roi: "strong highlight",
    dim_non_roi: "dim",
    show_focus_prompt: "prompt",
    recommend_zoom: "zoom recommendation",
  };
  return actions.map((a) => map[a] || a).join(", ");
}

function resolveAdaptationStatus(actions) {
  if (!actions || actions.length === 0) return "Idle — no intervention needed";
  return "Intervention triggered";
}

function resolveAdaptationDetail(actions) {
  if (!actions || actions.length === 0) {
    if (policyInfo && isRiskAtLeast(currentRiskLevel, policyInfo.intervention_min_risk)) {
      return "Armed — risk rising";
    }
    return "Monitoring — risk below threshold";
  }
  return `Actions: ${formatActions(actions)}`;
}

function isRiskAtLeast(level, threshold) {
  const order = { low: 0, medium: 1, high: 2 };
  return (order[level] ?? 0) >= (order[threshold] ?? 1);
}

function mapMouseToImage(event) {
  const rect = sliceImage.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    return { x: 0, y: 0 };
  }
  const scaleX = imageDims.width / rect.width;
  const scaleY = imageDims.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  return { x, y };
}

async function sendMouseGaze(event) {
  if (sourceSelect.value !== "mouse") return;
  const now = Date.now();
  if (now - lastMouseSent < 100) return;
  lastMouseSent = now;
  const mapped = mapMouseToImage(event);
  await fetch("/api/gaze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      point: {
        timestamp: now,
        x: mapped.x,
        y: mapped.y,
        source: "mouse",
        mode: modeSelect.value,
      },
    }),
  });
}

sliceImage.addEventListener("load", () => {
  drawRoi();
});

document.getElementById("image-stage").addEventListener("mousemove", sendMouseGaze);

toggleRoi.addEventListener("change", drawRoi);
toggleGaze.addEventListener("change", () => drawGaze(null));
toggleAdapt.addEventListener("change", () => pollState());
togglePredict.addEventListener("change", updateMode);
toggleAdapt.addEventListener("change", updateMode);
policySelect.addEventListener("change", updateMode);
messageMinRisk.addEventListener("change", updateMode);
interventionMinRisk.addEventListener("change", updateMode);
cooldownMs.addEventListener("change", updateMode);
messageHoldMs.addEventListener("change", updateMode);
silentLowRisk.addEventListener("change", updateMode);
toggleDebug.addEventListener("change", () => {
  debugPanel.style.display = toggleDebug.checked ? "block" : "none";
});

sourceSelect.addEventListener("change", updateMode);
modeSelect.addEventListener("change", updateMode);
caseSelect.addEventListener("change", () => {
  currentCase = caseSelect.value;
  sliceInput.value = 0;
  loadSlice();
});

loadSliceBtn.addEventListener("click", loadSlice);
resetButton.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
});

async function init() {
  await fetchCases();
  await loadSlice();
  const policyResponse = await fetch("/api/policy/info");
  policyInfo = await policyResponse.json();
  if (policyInfo && policyInfo.policy_mode) {
    policySelect.value = policyInfo.policy_mode;
    messageMinRisk.value = policyInfo.message_min_risk;
    interventionMinRisk.value = policyInfo.intervention_min_risk;
    cooldownMs.value = policyInfo.cooldown_ms;
    messageHoldMs.value = policyInfo.message_hold_ms;
    silentLowRisk.checked = policyInfo.silent_low_risk;
  }
  debugPanel.style.display = "none";
  setInterval(pollState, 500);
}

init();
