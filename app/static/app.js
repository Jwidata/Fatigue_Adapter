const caseSelect = document.getElementById("case-select");
const sliceInput = document.getElementById("slice-input");
const loadSliceBtn = document.getElementById("load-slice");
const sliceImage = document.getElementById("slice-image");
const roiCanvas = document.getElementById("roi-canvas");
const roiEditCanvas = document.getElementById("roi-edit-canvas");
const gazeCanvas = document.getElementById("gaze-canvas");
const toggleRoi = document.getElementById("toggle-roi");
const toggleRoiDebug = document.getElementById("toggle-roi-debug");
const toggleGaze = document.getElementById("toggle-gaze");
const togglePredict = document.getElementById("toggle-predict");
const toggleAdapt = document.getElementById("toggle-adapt");
const toggleDebug = document.getElementById("toggle-debug");
const toggleRoiEdit = document.getElementById("toggle-roi-edit");
const clearRoiBtn = document.getElementById("clear-roi");
const undoRoiBtn = document.getElementById("undo-roi");
const roiHint = document.getElementById("roi-hint");
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
const statusAlert = document.getElementById("status-alert");
const statusTobii = document.getElementById("status-tobii");
const statusMode = document.getElementById("status-mode");
const statusData = document.getElementById("status-data");
const riskFill = document.getElementById("risk-fill");
const predConfidenceBar = document.getElementById("pred-confidence-bar");
const metricCoverageGauge = document.getElementById("metric-coverage-gauge");
const metricErrorBar = document.getElementById("metric-error-bar");
const metricScanBar = document.getElementById("metric-scan-bar");
const predMode = document.getElementById("pred-mode");
const predActive = document.getElementById("pred-active");
const predExplain = document.getElementById("pred-explain");
const predStatus = document.getElementById("pred-status");
const predConfidence = document.getElementById("pred-confidence");
const predAccuracy = document.getElementById("pred-accuracy");
const modelBadge = document.getElementById("model-badge");
const modelName = document.getElementById("model-name");
const modelScore = document.getElementById("model-score");
const modelWithin = document.getElementById("model-within");
const modelEfficiency = document.getElementById("model-efficiency");
const modelDescription = document.getElementById("model-description");
const datasetBadge = document.getElementById("dataset-badge");
const datasetSummary = document.getElementById("dataset-summary");
const datasetFeatures = document.getElementById("dataset-features");
const datasetTarget = document.getElementById("dataset-target");
const datasetSamples = document.getElementById("dataset-samples");
const imageStage = document.getElementById("image-stage");
const imageOverlay = document.getElementById("image-overlay");
const debugPanel = document.getElementById("debug-panel");
const uploadFile = document.getElementById("upload-file");
const uploadCase = document.getElementById("upload-case");
const uploadSeries = document.getElementById("upload-series");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const mockTobiiButton = document.getElementById("mock-tobii");
const mockStatus = document.getElementById("mock-status");

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
let roiMaskCache = null;
let roiMaskDims = null;
let roiFallbackActive = false;
let lastRoiCenterCanvas = null;
let lastInRoi = false;
let predictorResults = null;
let datasetInfo = null;
let roiDragStart = null;
let roiDragCurrent = null;
let lastViewportSent = 0;
let currentRoiSource = "";
const roiUndoStack = [];
let mockTobiiTimer = null;

async function fetchCases() {
  const response = await fetch("/api/cases");
  const data = await response.json();
  caseSelect.innerHTML = "";
  if (!data.cases || data.cases.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No cases found";
    caseSelect.appendChild(option);
    currentCase = null;
    statusData.textContent = "Dataset: no cases loaded — check Data directory";
    loadSliceBtn.disabled = true;
    return;
  }
  data.cases.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.case_id;
    option.textContent = entry.case_id;
    caseSelect.appendChild(option);
  });
  currentCase = data.cases[0].case_id;
  caseSelect.value = currentCase;
  loadSliceBtn.disabled = false;
}

async function loadSlice() {
  if (!currentCase) return;
  currentSlice = parseInt(sliceInput.value, 10) || 0;
  const imgUrl = `/api/cases/${currentCase}/slices/${currentSlice}`;
  sliceImage.src = imgUrl;
  await updateMode();
  await fetchRoi();
}

async function fetchPredictorResults() {
  const response = await fetch("/api/predictor/results");
  const data = await response.json();
  predictorResults = data.status === "ok" ? data.results : null;
}

async function fetchDatasetSummary() {
  const response = await fetch("/api/dataset/summary");
  const data = await response.json();
  datasetInfo = data.status === "ok" ? data.summary : null;
  if (datasetInfo && datasetInfo.num_train_samples !== undefined) {
    statusData.textContent = `Dataset: ${datasetInfo.num_train_samples} train / ${datasetInfo.num_val_samples} val / ${datasetInfo.num_test_samples} test`;
  }
}

async function fetchRoi() {
  if (!currentCase) return;
  const response = await fetch(`/api/roi/${currentCase}/${currentSlice}`);
  const data = await response.json();
  currentRoi = data.rois || [];
  currentRoiSource = data.source || "";
  imageDims = { width: data.image_width, height: data.image_height };
  if (imageDims.width && imageDims.height) {
    resizeCanvas();
  }
  roiMaskCache = null;
  roiMaskDims = null;
  console.log("ROI payload", {
    image_width: data.image_width,
    image_height: data.image_height,
    rois: currentRoi.map((roi) => ({
      type: roi.type,
      bbox: roi.bbox,
      mask_shape: Array.isArray(roi.mask)
        ? [roi.mask.length, roi.mask[0] ? roi.mask[0].length : 0]
        : roi.mask && roi.mask.encoded
          ? "encoded"
          : null,
    })),
  });
  drawRoi();
  await sendViewport();
  updateRoiUndoState();
}

function resizeCanvas() {
  const rect = sliceImage.getBoundingClientRect();
  let width = rect.width;
  let height = rect.height;
  if (!width || !height) {
    const stageRect = imageStage.getBoundingClientRect();
    width = stageRect.width;
    height = stageRect.height;
  }
  if (!width || !height) {
    return;
  }
  roiCanvas.width = width;
  roiCanvas.height = height;
  roiEditCanvas.width = width;
  roiEditCanvas.height = height;
  gazeCanvas.width = width;
  gazeCanvas.height = height;
}

async function sendViewport() {
  if (!currentCase || !imageDims.width || !imageDims.height) return;
  const rect = sliceImage.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const now = Date.now();
  if (now - lastViewportSent < 250) return;
  lastViewportSent = now;
  await fetch("/api/viewport", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: currentCase,
      slice_id: currentSlice,
      image_left: rect.left,
      image_top: rect.top,
      image_width: rect.width,
      image_height: rect.height,
      image_pixel_width: imageDims.width,
      image_pixel_height: imageDims.height,
      screen_width: window.innerWidth,
      screen_height: window.innerHeight,
      timestamp: now,
    }),
  });
}

async function sendMockTobiiSample(xNorm, yNorm) {
  if (!currentCase) return;
  await fetch("/api/gaze_display", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      timestamp: Date.now(),
      x: xNorm,
      y: yNorm,
      screen_width: window.innerWidth,
      screen_height: window.innerHeight,
      case_id: currentCase,
      slice_id: currentSlice,
      source: "tobii",
      mode: modeSelect.value,
      normalized: true,
    }),
  });
}

function toggleMockTobii() {
  if (mockTobiiTimer) {
    clearInterval(mockTobiiTimer);
    mockTobiiTimer = null;
    mockTobiiButton.textContent = "Start mock stream";
    mockStatus.textContent = "Status: idle";
    return;
  }
  mockTobiiButton.textContent = "Stop mock stream";
  mockStatus.textContent = "Status: streaming";
  const start = Date.now();
  mockTobiiTimer = setInterval(() => {
    const t = (Date.now() - start) / 1000;
    const xNorm = 0.5 + 0.12 * Math.sin(t * 1.6);
    const yNorm = 0.5 + 0.12 * Math.cos(t * 1.2);
    sendViewport();
    sendMockTobiiSample(Math.max(0, Math.min(1, xNorm)), Math.max(0, Math.min(1, yNorm)));
  }, 120);
}

function getScale() {
  if (!imageDims.width || !imageDims.height) {
    return { scaleX: 1, scaleY: 1 };
  }
  return {
    scaleX: roiCanvas.width / imageDims.width,
    scaleY: roiCanvas.height / imageDims.height,
  };
}

function normalizeBbox(bbox) {
  if (!bbox) return null;
  const x = bbox.x ?? 0;
  const y = bbox.y ?? 0;
  const w = bbox.w ?? bbox.width ?? 0;
  const h = bbox.h ?? bbox.height ?? 0;
  const isNormalized = Math.max(x, y, w, h) <= 1.0;
  return { x, y, w, h, isNormalized };
}

function getFallbackRoi() {
  const size = 120;
  const cx = imageDims.width / 2;
  const cy = imageDims.height / 2;
  return [{ type: "bbox", bbox: { x: cx - size / 2, y: cy - size / 2, width: size, height: size } }];
}

function buildMaskCache(mask) {
  if (!Array.isArray(mask) || mask.length === 0) return null;
  const rows = mask.length;
  const cols = Array.isArray(mask[0]) ? mask[0].length : 0;
  if (!rows || !cols) return null;
  const offscreen = document.createElement("canvas");
  offscreen.width = cols;
  offscreen.height = rows;
  const ctx = offscreen.getContext("2d");
  const imageData = ctx.createImageData(cols, rows);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const idx = (y * cols + x) * 4;
      const value = mask[y][x] ? 255 : 0;
      imageData.data[idx] = 255;
      imageData.data[idx + 1] = 60;
      imageData.data[idx + 2] = 60;
      imageData.data[idx + 3] = value;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return { canvas: offscreen, rows, cols };
}

function drawRoi() {
  if (!imageDims.width || !imageDims.height) return;
  resizeCanvas();
  if (!roiCanvas.width || !roiCanvas.height) return;
  const ctx = roiCanvas.getContext("2d");
  ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  const showRoi = toggleRoi.checked || toggleRoiDebug.checked;
  if (!showRoi) return;
  let rois = currentRoi;
  if (!rois.length) {
    console.error("ROI missing");
    rois = getFallbackRoi();
    roiFallbackActive = true;
  } else {
    roiFallbackActive = false;
  }
  const { scaleX, scaleY } = getScale();
  if (!scaleX || !scaleY) return;
  lastRoiCenterCanvas = null;
  ctx.strokeStyle = "rgba(255, 59, 47, 0.95)";
  ctx.lineWidth = 3;
  ctx.fillStyle = "rgba(255, 59, 47, 0.25)";
  rois.forEach((roi) => {
    if (roi.mask && Array.isArray(roi.mask)) {
      if (!roiMaskCache) {
        roiMaskCache = buildMaskCache(roi.mask);
        roiMaskDims = roiMaskCache ? { width: roiMaskCache.cols, height: roiMaskCache.rows } : null;
      }
      if (roiMaskCache) {
        ctx.globalAlpha = 0.3;
        ctx.drawImage(roiMaskCache.canvas, 0, 0, roiCanvas.width, roiCanvas.height);
        ctx.globalAlpha = 1.0;
      }
    }
    if (roi.type === "bbox" && roi.bbox) {
      const normalized = normalizeBbox(roi.bbox);
      if (!normalized) return;
      const imageX = normalized.isNormalized ? normalized.x * imageDims.width : normalized.x;
      const imageY = normalized.isNormalized ? normalized.y * imageDims.height : normalized.y;
      const imageW = normalized.isNormalized ? normalized.w * imageDims.width : normalized.w;
      const imageH = normalized.isNormalized ? normalized.h * imageDims.height : normalized.h;
      const x = imageX * scaleX;
      const y = imageY * scaleY;
      const w = imageW * scaleX;
      const h = imageH * scaleY;
      const area = Math.max(0, w) * Math.max(0, h);
      console.log("ROI render", {
        bbox: { x: imageX, y: imageY, w: imageW, h: imageH },
        image_size: { width: imageDims.width, height: imageDims.height },
        canvas_size: { width: roiCanvas.width, height: roiCanvas.height },
        area,
      });
      if (!area) {
        console.error("ROI area is zero");
      }
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
      ctx.fillRect(x, Math.max(0, y - 22), 170, 20);
      ctx.fillStyle = "#fff";
      ctx.font = "12px 'Palatino Linotype', serif";
      ctx.fillText(roiFallbackActive ? "DEBUG ROI" : "Critical Region", x + 6, Math.max(14, y - 8));
      lastRoiCenterCanvas = { x: x + w / 2, y: y + h / 2 };
      if (toggleRoiDebug.checked) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(x, y + h + 4, 220, 18);
        ctx.fillStyle = "#fff";
        ctx.font = "11px 'Palatino Linotype', serif";
        ctx.fillText(`x:${x.toFixed(1)} y:${y.toFixed(1)} w:${w.toFixed(1)} h:${h.toFixed(1)}`, x + 6, y + h + 18);
      }
      ctx.fillStyle = "rgba(255, 59, 47, 0.2)";
    }
    if (!roi) {
      console.error("ROI missing");
    }
  });
  if (toggleRoi.checked || toggleRoiDebug.checked) {
    roiCanvas.style.animation = "roiPulse 2.6s ease-in-out infinite";
  } else {
    roiCanvas.style.animation = "none";
  }
}

function drawRoiEdit() {
  if (!roiEditCanvas.width || !roiEditCanvas.height) return;
  const ctx = roiEditCanvas.getContext("2d");
  ctx.clearRect(0, 0, roiEditCanvas.width, roiEditCanvas.height);
  if (!roiDragStart || !roiDragCurrent) return;
  const x = Math.min(roiDragStart.x, roiDragCurrent.x);
  const y = Math.min(roiDragStart.y, roiDragCurrent.y);
  const w = Math.abs(roiDragStart.x - roiDragCurrent.x);
  const h = Math.abs(roiDragStart.y - roiDragCurrent.y);
  ctx.strokeStyle = "rgba(255, 59, 47, 0.9)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255, 59, 47, 0.2)";
  ctx.fillRect(x, y, w, h);
}

function mapCanvasToImage(point) {
  if (!imageDims.width || !imageDims.height) return null;
  const { scaleX, scaleY } = getScale();
  if (!scaleX || !scaleY) return null;
  return {
    x: point.x / scaleX,
    y: point.y / scaleY,
  };
}

function updateRoiUndoState() {
  undoRoiBtn.disabled = roiUndoStack.length === 0;
}

async function applyRoiOverride(bbox, label = "User ROI") {
  if (!currentCase) return;
  await fetch("/api/roi/override", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: currentCase,
      slice_id: currentSlice,
      bbox,
      label,
      priority: 0.85,
    }),
  });
  await fetchRoi();
}

async function setRoiOverrideFromCanvas() {
  if (!roiDragStart || !roiDragCurrent || !currentCase) return;
  const start = mapCanvasToImage(roiDragStart);
  const end = mapCanvasToImage(roiDragCurrent);
  if (!start || !end) return;
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const w = Math.abs(start.x - end.x);
  const h = Math.abs(start.y - end.y);
  if (w < 5 || h < 5) return;
  const bbox = {
    x: x / imageDims.width,
    y: y / imageDims.height,
    w: w / imageDims.width,
    h: h / imageDims.height,
  };
  if (currentRoiSource === "user" && currentRoi.length && currentRoi[0].bbox) {
    roiUndoStack.push(currentRoi[0].bbox);
  }
  await applyRoiOverride(bbox);
  updateRoiUndoState();
}

function drawGaze(point, prediction) {
  if (!imageDims.width || !imageDims.height) return;
  resizeCanvas();
  if (!gazeCanvas.width || !gazeCanvas.height) return;
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

  if (lastRoiCenterCanvas && toggleRoi.checked && !lastInRoi) {
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(lastRoiCenterCanvas.x, lastRoiCenterCanvas.y);
    ctx.strokeStyle = "rgba(255, 59, 47, 0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
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

  const statusMap = {
    at_risk: "🔴 Critical",
    drifting_attention: "🟡 Attention",
    low_roi_engagement: "🟡 Attention",
    normal: "🟢 Normal",
  };
  statusMain.textContent = statusMap[data.state] || `State: ${data.state}`;

  const inRoi = isGazeInRoi(data.latest_gaze);
  lastInRoi = inRoi;
  if (!data.latest_gaze) {
    statusAttention.textContent = "--";
    statusAlert.textContent = "Alert: --";
    statusAlert.classList.remove("alert");
  } else if (inRoi) {
    statusAttention.innerHTML = `<span class="indicator-good">✔ Good focus — continue</span>`;
    statusAlert.textContent = "Alert: none";
    statusAlert.classList.remove("alert");
  } else {
    statusAttention.innerHTML = `<span class="indicator-bad">⚠ Shift attention → red region</span>`;
    statusAlert.textContent = "Alert: gaze outside ROI";
    statusAlert.classList.add("alert");
  }
  updateTobiiStatus(data.latest_gaze);
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
  const roiVisible = (currentRoi.length > 0 || roiFallbackActive) && (toggleRoi.checked || toggleRoiDebug.checked);
  statusRoi.textContent = `Critical region: ${roiVisible ? "visible" : "hidden"}`;

  updateRiskBar(data.risk ? data.risk.risk_level : "low");
  updateMetricVisuals(data.metrics);
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
  if (!imageDims.width || !imageDims.height) {
    if (sliceImage.naturalWidth && sliceImage.naturalHeight) {
      imageDims = { width: sliceImage.naturalWidth, height: sliceImage.naturalHeight };
      resizeCanvas();
    }
  }
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
  if (info.confidence !== null && info.confidence !== undefined) {
    const pct = Math.max(0, Math.min(1, info.confidence)) * 100;
    predConfidenceBar.style.width = `${pct}%`;
  } else {
    predConfidenceBar.style.width = "0%";
  }
  predAccuracy.textContent = `Accuracy: ${info.accuracy_note}`;

  if (toggleDebug.checked && info.eval_stats) {
    if (info.eval_stats.count >= 10) {
      metrics.predMean.textContent = `Prediction error mean: ${info.eval_stats.mean_error_px.toFixed(1)} px`;
      metrics.predMedian.textContent = `Prediction error median: ${info.eval_stats.median_error_px.toFixed(1)} px`;
      metrics.predWithin.textContent = `Within 50px: ${info.eval_stats.within_50_px_pct.toFixed(1)}%`;
    } else {
      metrics.predMean.textContent = "Prediction error mean: Not enough samples yet";
      metrics.predMedian.textContent = "Prediction error median: Not enough samples yet";
      metrics.predWithin.textContent = "Prediction within 25/50 px: Not enough samples yet";
    }
  } else if (toggleDebug.checked) {
    metrics.predMean.textContent = "Prediction error mean: Not enough samples yet";
    metrics.predMedian.textContent = "Prediction error median: Not enough samples yet";
    metrics.predWithin.textContent = "Within 50px: Not enough samples yet";
  }
  updateModelCard(info);
}

function updateModelCard(info) {
  const active = info.active_predictor || "--";
  const efficiency = estimateEfficiency(active);
  modelBadge.textContent = active.toUpperCase();
  modelName.textContent = `Model: ${active}`;
  modelEfficiency.textContent = `Efficiency: ${efficiency}`;
  modelDescription.textContent = `Description: ${modelDescriptionFor(active)}`;
  const score = predictorResults && predictorResults[active] ? predictorResults[active] : null;
  if (score && score.status === "evaluated") {
    modelScore.textContent = `Score (mean error): ${score.mean_error.toFixed(1)} px`;
    modelWithin.textContent = `Within 50px: ${score.within_50.toFixed(1)}%`;
  } else {
    modelScore.textContent = "Score (mean error): --";
    modelWithin.textContent = "Within 50px: --";
  }
}

function updateDatasetCard() {
  if (!datasetInfo) {
    datasetBadge.textContent = "MISSING";
    datasetSummary.textContent = "Summary: dataset summary not found";
    datasetFeatures.textContent = "Features: --";
    datasetTarget.textContent = "Target: --";
    datasetSamples.textContent = "Samples: --";
    return;
  }
  datasetBadge.textContent = "ACTIVE";
  datasetSummary.textContent = `Summary: seq ${datasetInfo.sequence_len}, horizon ${datasetInfo.horizon || 1}`;
  datasetFeatures.textContent = `Features: ${datasetInfo.feature_names ? datasetInfo.feature_names.join(", ") : "--"}`;
  datasetTarget.textContent = `Target: ${datasetInfo.target || "--"}`;
  datasetSamples.textContent = `Samples: ${datasetInfo.num_train_samples}/${datasetInfo.num_val_samples}/${datasetInfo.num_test_samples}`;
}

function estimateEfficiency(model) {
  const map = {
    xgboost: "fast (CPU)",
    gru: "medium",
    transformer: "slower",
    lstm: "medium",
    temporal_cnn: "fast",
    heuristic: "instant",
    constant_velocity: "instant",
  };
  return map[model] || "unknown";
}

function modelDescriptionFor(model) {
  const map = {
    xgboost: "Tree-based regressor optimized for latency-sensitive predictions.",
    gru: "Sequence model capturing temporal gaze dynamics.",
    transformer: "Attention-based model for complex gaze patterns.",
    lstm: "Sequence model with long-range temporal memory.",
    temporal_cnn: "Temporal convolution for short-range motion cues.",
    heuristic: "Velocity baseline when models are unavailable.",
    constant_velocity: "Constant velocity baseline for next-step prediction.",
  };
  return map[model] || "Model description unavailable.";
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
  if (!actions || actions.length === 0) return "✔ Monitoring";
  if (actions.includes("highlight_roi")) return "⚠ Focus here";
  if (actions.includes("strong_highlight_roi") || actions.includes("dim_non_roi")) return "🔴 Intervention triggered";
  return "⚠ Suggesting focus";
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

function updateRiskBar(level) {
  const map = { low: 20, medium: 55, high: 85 };
  const pct = map[level] || 10;
  riskFill.style.width = `${pct}%`;
}

function updateMetricVisuals(metricsData) {
  if (!metricsData) return;
  const coverage = metricsData.roi_coverage_pct ?? 0;
  const scan = metricsData.scan_coverage_pct ?? 0;
  const error = metricsData.dispersion ?? 0;
  const coverageDeg = Math.max(0, Math.min(coverage, 100)) * 3.6;
  const coverageColor = coverage < 30 ? "#ff3b2f" : coverage < 70 ? "#f4b000" : "#2f9e44";
  metricCoverageGauge.style.background = `conic-gradient(${coverageColor} 0deg ${coverageDeg}deg, rgba(39, 76, 89, 0.1) ${coverageDeg}deg 360deg)`;
  metricScanBar.style.width = `${Math.max(0, Math.min(scan, 100))}%`;
  const errorPct = Math.min(error * 100, 100);
  metricErrorBar.style.width = `${errorPct}%`;
}

function updateTobiiStatus(gazePoint) {
  if (sourceSelect.value !== "tobii") {
    statusTobii.textContent = "Tobii: inactive";
    statusTobii.classList.remove("alert");
    return;
  }
  if (!gazePoint || gazePoint.source !== "tobii") {
    statusTobii.textContent = "Tobii: waiting";
    statusTobii.classList.add("alert");
    return;
  }
  const ageMs = Date.now() - gazePoint.timestamp;
  if (ageMs > 1500) {
    statusTobii.textContent = "Tobii: stale";
    statusTobii.classList.add("alert");
    return;
  }
  statusTobii.textContent = "Tobii: live";
  statusTobii.classList.remove("alert");
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

function mapPointerToCanvas(event) {
  const rect = imageStage.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    return { x: 0, y: 0 };
  }
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return {
    x: Math.max(0, Math.min(roiEditCanvas.width, x)),
    y: Math.max(0, Math.min(roiEditCanvas.height, y)),
  };
}

function startRoiDrag(event) {
  if (!toggleRoiEdit.checked) return;
  roiDragStart = mapPointerToCanvas(event);
  roiDragCurrent = { ...roiDragStart };
  drawRoiEdit();
}

function moveRoiDrag(event) {
  if (!toggleRoiEdit.checked || !roiDragStart) return;
  roiDragCurrent = mapPointerToCanvas(event);
  drawRoiEdit();
}

async function endRoiDrag() {
  if (!toggleRoiEdit.checked || !roiDragStart) return;
  await setRoiOverrideFromCanvas();
  roiDragStart = null;
  roiDragCurrent = null;
  drawRoiEdit();
}

async function sendMouseGaze(event) {
  if (sourceSelect.value !== "mouse" || toggleRoiEdit.checked) return;
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
  if (sliceImage.naturalWidth && sliceImage.naturalHeight) {
    imageDims = { width: sliceImage.naturalWidth, height: sliceImage.naturalHeight };
  }
  resizeCanvas();
  drawRoi();
  sendViewport();
});

sliceImage.addEventListener("error", () => {
  resizeCanvas();
  drawRoi();
});

window.addEventListener("resize", () => {
  resizeCanvas();
  drawRoi();
  sendViewport();
});

document.getElementById("image-stage").addEventListener("mousemove", sendMouseGaze);
imageStage.addEventListener("mousedown", startRoiDrag);
imageStage.addEventListener("mousemove", moveRoiDrag);
imageStage.addEventListener("mouseup", endRoiDrag);
imageStage.addEventListener("mouseleave", endRoiDrag);

toggleRoi.addEventListener("change", drawRoi);
toggleRoiDebug.addEventListener("change", drawRoi);
toggleGaze.addEventListener("change", () => drawGaze(null));
toggleAdapt.addEventListener("change", () => pollState());
toggleRoiEdit.addEventListener("change", () => {
  if (toggleRoiEdit.checked) {
    imageStage.classList.add("editing");
    roiHint.textContent = "Drag on image to define region";
  } else {
    imageStage.classList.remove("editing");
    roiDragStart = null;
    roiDragCurrent = null;
    drawRoiEdit();
    roiHint.textContent = "ROI editing disabled";
  }
});
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

clearRoiBtn.addEventListener("click", async () => {
  if (!currentCase) return;
  await fetch(`/api/roi/override/${currentCase}/${currentSlice}`, { method: "DELETE" });
  roiUndoStack.length = 0;
  await fetchRoi();
  updateRoiUndoState();
});

undoRoiBtn.addEventListener("click", async () => {
  if (roiUndoStack.length === 0) return;
  const prev = roiUndoStack.pop();
  if (prev) {
    await applyRoiOverride(prev, "User ROI (undo)");
  }
  updateRoiUndoState();
});

uploadButton.addEventListener("click", async () => {
  if (!uploadFile.files || uploadFile.files.length === 0) {
    uploadStatus.textContent = "Status: select a file";
    return;
  }
  const form = new FormData();
  form.append("file", uploadFile.files[0]);
  if (uploadCase.value) form.append("case_id", uploadCase.value);
  if (uploadSeries.value) form.append("series_id", uploadSeries.value);
  uploadStatus.textContent = "Status: uploading...";
  const response = await fetch("/api/upload", { method: "POST", body: form });
  if (!response.ok) {
    uploadStatus.textContent = "Status: upload failed";
    return;
  }
  const data = await response.json();
  uploadStatus.textContent = `Status: uploaded ${data.case_id}`;
  await fetchCases();
});

mockTobiiButton.addEventListener("click", () => {
  sourceSelect.value = "tobii";
  toggleMockTobii();
});

async function init() {
  await fetchCases();
  await fetchPredictorResults();
  await fetchDatasetSummary();
  updateDatasetCard();
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
  roiHint.textContent = "ROI editing disabled";
  updateRoiUndoState();
  setInterval(pollState, 500);
  requestAnimationFrame(renderLoop);
}

init();

function renderLoop() {
  drawRoi();
  drawRoiEdit();
  drawGaze(latestGaze, latestPrediction);
  requestAnimationFrame(renderLoop);
}
