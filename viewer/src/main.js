import * as ort from "onnxruntime-web";

const DEFAULT_MODEL_PATH = "./segmentation_model_timm.onnx";
const DEFAULT_INPUT_SIZE = { rows: 320, cols: 256 };

const state = {
  session: null,
  imageBitmap: null,
  imageUrl: "",
  imageName: "",
  overlayCanvas: null,
  overlayVisible: true,
  modelSource: DEFAULT_MODEL_PATH,
  modelLabel: "Bundled AI model",
  modelFile: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  diagnosticsVisible: false,
  controlsCollapsed: false,
  activePointerId: null,
};

const app = document.querySelector("#app");
app.innerHTML = `
  <div class="page-shell">
    <section class="hero compact-hero">
      <div>
        <p class="eyebrow">Clinical AI Viewer</p>
        <h1>Review image and run AI analysis.</h1>
        <p class="lede">Load an AI model and image to generate a segmentation overlay for clinical review.</p>
      </div>
      <div class="status-card" id="modelStatus">AI model not loaded.</div>
    </section>

    <section class="workspace medical-layout" id="workspaceLayout">
      <article class="panel viewer-panel">
        <div class="panel-head viewer-head">
          <div>
            <h2>Main Viewer</h2>
            <span id="viewerHint">Load an AI model and image to begin.</span>
          </div>
          <div class="viewer-actions">
            <span class="zoom-pill" id="zoomLevel">100%</span>
            <button id="zoomOutButton" class="secondary" disabled>Zoom -</button>
            <button id="zoomResetButton" class="secondary" disabled>Reset</button>
            <button id="zoomInButton" class="secondary" disabled>Zoom +</button>
          </div>
        </div>
        <div class="viewer-stage checkerboard" id="viewerStage">
          <canvas id="viewerCanvas"></canvas>
          <div class="viewer-minimap hidden" id="miniMapWrap" aria-hidden="true">
            <canvas id="miniMapCanvas" width="180" height="118"></canvas>
          </div>
        </div>
      </article>

      <aside class="panel toolbar-panel" id="toolbarPanel">
        <div class="panel-head">
          <h2>Controls</h2>
          <div class="toolbar-head-actions">
            <button id="toggleControlsButton" class="secondary">Collapse</button>
          </div>
        </div>

        <div class="tool-section">
          <div class="tool-label">AI Model</div>
          <div class="model-selector-group">
            <label for="modelSelector" class="select-label">Choose model:</label>
            <select id="modelSelector" class="model-selector">
              <option value="bundled">Bundled AI model</option>
              <option value="upload">Upload custom model</option>
            </select>
          </div>
          <div id="modelUploadSection" class="hidden">
            <label class="upload-card" for="modelInput">
              <span class="upload-title">Select model file</span>
              <span class="upload-copy">.onnx file</span>
              <input id="modelInput" type="file" accept=".onnx,application/octet-stream" />
            </label>
          </div>
          <div class="button-group">
            <button id="loadModelButton" class="primary">Load AI model</button>
          </div>
          <div class="metrics tool-metrics" id="modelMeta">Current model: bundled AI model</div>
        </div>

        <div class="tool-section">
          <div class="tool-label">Image</div>
          <label class="upload-card" for="imageInput">
            <span class="upload-title">Select image</span>
            <span class="upload-copy">PNG, JPG, JPEG</span>
            <input id="imageInput" type="file" accept="image/png,image/jpeg" />
          </label>
          <div class="button-group">
            <button id="runButton" class="primary" disabled>Run analysis</button>
          </div>
          <div class="metrics tool-metrics" id="imageMeta">No image loaded.</div>
        </div>

        <div class="tool-section">
          <div class="tool-label">Overlay</div>
          <label class="toggle-control" for="overlayVisible">
            <input id="overlayVisible" type="checkbox" checked />
            <span>Show segmentation</span>
          </label>
          <label class="overlay-control" for="overlayOpacity">Overlay opacity</label>
          <input id="overlayOpacity" type="range" min="0" max="100" value="55" />
          <span id="overlayOpacityValue">55%</span>
        </div>

        <div class="tool-section">
          <div class="tool-label">Advanced</div>
          <button id="diagnosticsToggleButton" class="secondary">Show diagnostics</button>
          <div id="diagnosticsPanel" class="diagnostics-panel hidden" aria-hidden="true">
            <div class="panel-head diagnostics-head">
              <h3>Diagnostics</h3>
              <button id="clearLogsButton" class="secondary">Clear</button>
            </div>
            <pre id="logOutput" class="log-output">Diagnostics are ready.</pre>
          </div>
        </div>
      </aside>
    </section>
  </div>
`;

const imageInput = document.querySelector("#imageInput");
const modelInput = document.querySelector("#modelInput");
const runButton = document.querySelector("#runButton");
const viewerStage = document.querySelector("#viewerStage");
const loadModelButton = document.querySelector("#loadModelButton");
const modelSelector = document.querySelector("#modelSelector");
const modelUploadSection = document.querySelector("#modelUploadSection");
const clearLogsButton = document.querySelector("#clearLogsButton");
const diagnosticsToggleButton = document.querySelector("#diagnosticsToggleButton");
const diagnosticsPanel = document.querySelector("#diagnosticsPanel");
const toggleControlsButton = document.querySelector("#toggleControlsButton");
const workspaceLayout = document.querySelector("#workspaceLayout");
const toolbarPanel = document.querySelector("#toolbarPanel");
const sourceCanvas = document.createElement("canvas");
const viewerCanvas = document.querySelector("#viewerCanvas");
const zoomLevel = document.querySelector("#zoomLevel");
const miniMapWrap = document.querySelector("#miniMapWrap");
const miniMapCanvas = document.querySelector("#miniMapCanvas");
const modelStatus = document.querySelector("#modelStatus");
const modelMeta = document.querySelector("#modelMeta");
const imageMeta = document.querySelector("#imageMeta");
const logOutput = document.querySelector("#logOutput");
const viewerHint = document.querySelector("#viewerHint");
const overlayVisible = document.querySelector("#overlayVisible");
const overlayOpacity = document.querySelector("#overlayOpacity");
const overlayOpacityValue = document.querySelector("#overlayOpacityValue");
const zoomInButton = document.querySelector("#zoomInButton");
const zoomOutButton = document.querySelector("#zoomOutButton");
const zoomResetButton = document.querySelector("#zoomResetButton");

const style = document.createElement("style");
style.textContent = `
  :root {
    color-scheme: light;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    --bg: #f4efe7;
    --card: rgba(255, 252, 247, 0.92);
    --ink: #1e293b;
    --muted: #5b6472;
    --line: rgba(30, 41, 59, 0.12);
    --accent: #0f766e;
    --accent-strong: #115e59;
    --warm: #d97706;
    --danger: #b91c1c;
    --shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background:
      radial-gradient(circle at top left, rgba(217, 119, 6, 0.16), transparent 24%),
      radial-gradient(circle at top right, rgba(15, 118, 110, 0.18), transparent 22%),
      linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
    color: var(--ink);
  }

  code {
    font-family: "SFMono-Regular", Consolas, monospace;
    background: rgba(15, 23, 42, 0.06);
    padding: 0.1rem 0.35rem;
    border-radius: 0.35rem;
  }

  button, input { font: inherit; }

  .page-shell {
    width: min(1280px, calc(100% - 32px));
    margin: 0 auto;
    padding: 32px 0 48px;
  }

  .hero, .panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 24px;
    box-shadow: var(--shadow);
  }

  .hero {
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 16px;
    padding: 18px 22px;
    margin-bottom: 20px;
  }

  .compact-hero h1 {
    max-width: 26ch;
    font-size: clamp(1.5rem, 2.4vw, 2.2rem);
    line-height: 1.05;
  }

  .eyebrow {
    margin: 0 0 12px;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--warm);
    font-weight: 700;
  }

  h1, h2 { margin: 0; }
  .lede { color: var(--muted); line-height: 1.6; margin: 8px 0 0; }

  .status-card {
    align-self: stretch;
    border-radius: 20px;
    padding: 14px 16px;
    background: linear-gradient(180deg, rgba(15, 118, 110, 0.09), rgba(15, 118, 110, 0.02));
    border: 1px solid rgba(15, 118, 110, 0.16);
    white-space: pre-line;
    font-weight: 600;
  }


  .upload-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.7);
    border: 1px dashed rgba(15, 118, 110, 0.3);
    cursor: pointer;
    transition: background 140ms ease, border-color 140ms ease;
  }

  .upload-card:hover {
    background: rgba(255,255,255,0.85);
    border-color: var(--accent);
  }

  .upload-title { font-weight: 700; }
  .upload-copy, .metrics, .panel-head span { color: var(--muted); }
  .metrics { overflow-wrap: anywhere; word-break: break-word; }
  .upload-card input { display: none; }

  button {
    border: 0;
    border-radius: 999px;
    padding: 14px 20px;
    cursor: pointer;
    transition: transform 140ms ease, opacity 140ms ease, background 140ms ease;
  }

  button {
    font-size: 14px;
    font-weight: 500;
  }

  button:hover:not(:disabled) { transform: translateY(-1px); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .primary { background: var(--accent); color: white; }
  .primary:hover:not(:disabled) { background: var(--accent-strong); }
  .secondary { background: #fff; color: var(--ink); border: 1px solid var(--line); }
  .secondary:hover:not(:disabled) { background: rgba(255, 255, 255, 0.95); border-color: var(--accent); }

  .workspace {
    display: grid;
    grid-template-columns: minmax(0, 2.3fr) minmax(320px, 0.7fr);
    gap: 20px;
    align-items: start;
  }

  .workspace.controls-collapsed {
    grid-template-columns: minmax(0, 1fr) 84px;
  }

  .medical-layout {
    margin-bottom: 20px;
  }

  .viewer-panel {
    padding: 18px;
  }

  .viewer-head {
    align-items: center;
  }

  .viewer-stage {
    min-height: 78vh;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid var(--line);
    position: relative;
    cursor: grab;
    touch-action: none;
    padding: 0;
  }

  .viewer-stage.dragging {
    cursor: grabbing;
  }

  .toolbar-panel {
    padding: 18px;
    position: sticky;
    top: 20px;
    min-width: 0;
  }

  .toolbar-panel.collapsed {
    padding: 12px;
  }

  .toolbar-panel.collapsed .tool-section,
  .toolbar-panel.collapsed .panel-head h2 {
    display: none;
  }

  .toolbar-head-actions {
    display: flex;
    justify-content: flex-end;
    width: 100%;
  }

  .viewer-actions {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  .zoom-pill {
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 700;
    color: #0f172a;
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid var(--line);
  }

  .tool-section {
    display: grid;
    gap: 12px;
    padding: 14px 0;
    border-top: 1px solid var(--line);
    min-width: 0;
  }

  .button-group {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    width: 100%;
    min-width: 0;
  }

  .button-group button {
    width: 100%;
    max-width: 100%;
    min-width: 0;
  }

  .model-selector-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .select-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--ink);
  }

  .model-selector {
    width: 100%;
    min-width: 0;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.8);
    color: var(--ink);
    font-size: 14px;
    cursor: pointer;
    transition: border-color 140ms ease, background 140ms ease;
  }

  .model-selector:hover {
    background: rgba(255, 255, 255, 0.95);
    border-color: var(--accent);
  }

  .model-selector:focus {
    outline: none;
    border-color: var(--accent);
    background: white;
  }

  .hidden {
    display: none;
  }

  .tool-section:first-of-type {
    border-top: 0;
    padding-top: 0;
  }

  .tool-label {
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--warm);
    font-weight: 700;
  }

  .tool-metrics {
    line-height: 1.6;
  }

  .overlay-control,
  .toggle-control {
    font-weight: 600;
  }

  .toggle-control {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  .toggle-control input {
    accent-color: var(--accent);
  }

  .panel { padding: 18px; }
  .panel-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: baseline;
    margin-bottom: 14px;
  }


  .checkerboard {
    background-image:
      linear-gradient(45deg, rgba(15, 23, 42, 0.04) 25%, transparent 25%),
      linear-gradient(-45deg, rgba(15, 23, 42, 0.04) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, rgba(15, 23, 42, 0.04) 75%),
      linear-gradient(-45deg, transparent 75%, rgba(15, 23, 42, 0.04) 75%);
    background-size: 22px 22px;
    background-position: 0 0, 0 11px, 11px -11px, -11px 0;
  }

  canvas {
    display: block;
  }

  #viewerCanvas {
    position: absolute;
    top: 0;
    left: 0;
    transform-origin: top left;
    max-width: none;
    max-height: none;
    will-change: transform;
  }

  .viewer-minimap {
    position: absolute;
    right: 12px;
    bottom: 12px;
    background: rgba(255, 255, 255, 0.93);
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.16);
  }

  .viewer-minimap.hidden {
    display: none;
  }

  #miniMapCanvas {
    width: 180px;
    height: 118px;
  }

  .log-output {
    margin: 0;
    min-height: 140px;
    max-height: 220px;
    overflow: auto;
    padding: 16px;
    border-radius: 16px;
    background: #111827;
    color: #e5e7eb;
    font: 13px/1.55 "SFMono-Regular", Consolas, monospace;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .diagnostics-panel {
    display: grid;
    gap: 10px;
  }

  .diagnostics-panel.hidden {
    display: none;
  }

  .diagnostics-head {
    margin-bottom: 0;
  }

  @media (max-width: 960px) {
    .hero, .workspace { grid-template-columns: 1fr; }
    .viewer-stage { min-height: 320px; }
    .toolbar-panel { position: static; }
    .workspace.controls-collapsed {
      grid-template-columns: 1fr;
    }

    .toolbar-panel.collapsed {
      padding: 18px;
    }

    .toolbar-panel.collapsed .tool-section,
    .toolbar-panel.collapsed .panel-head h2 {
      display: initial;
    }

    .toolbar-head-actions,
    .viewer-actions {
      width: 100%;
      justify-content: flex-start;
      flex-wrap: wrap;
    }

    .viewer-actions button {
      flex: 1 1 auto;
    }

    .viewer-minimap {
      transform: scale(0.92);
      transform-origin: bottom right;
    }
  }
`;
document.head.append(style);

imageInput.addEventListener("change", handleImageSelection);
overlayVisible.addEventListener("change", () => {
  state.overlayVisible = overlayVisible.checked;
  overlayOpacity.disabled = !state.overlayVisible;
  redrawViewer();
});
overlayOpacity.addEventListener("input", () => {
  overlayOpacityValue.textContent = `${overlayOpacity.value}%`;
  redrawViewer();
});
modelInput.addEventListener("change", handleModelSelection);
runButton.addEventListener("click", runInference);
loadModelButton.addEventListener("click", loadSelectedModel);
modelSelector.addEventListener("change", handleModelSelectorChange);
clearLogsButton.addEventListener("click", () => {
  logOutput.textContent = "Diagnostics cleared.";
});
diagnosticsToggleButton.addEventListener("click", toggleDiagnostics);
toggleControlsButton.addEventListener("click", toggleControls);
zoomInButton.addEventListener("click", () => changeZoom(0.2));
zoomOutButton.addEventListener("click", () => changeZoom(-0.2));
zoomResetButton.addEventListener("click", resetZoom);
viewerStage.addEventListener("wheel", handleZoomWheel, { passive: false });
viewerStage.addEventListener("pointerdown", startPan);
viewerStage.addEventListener("pointermove", movePan);
viewerStage.addEventListener("pointerup", endPan);
viewerStage.addEventListener("pointercancel", endPan);
viewerStage.addEventListener("pointerleave", endPan);
window.addEventListener("resize", () => {
  if (state.imageBitmap) {
    resetZoom();
  }
});

initializeRuntime();
setZoomControlsEnabled(false);
logLine(`Viewer ready. Default model path: ${DEFAULT_MODEL_PATH}`);

async function initializeRuntime() {
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = 1;

  if (window.location.port === "5173") {
    ort.env.wasm.wasmPaths = {
      wasm: "./ort-wasm-simd-threaded.wasm",
    };
    logLine("ONNX Runtime configured with Vite dev wasm path override.");
  } else {
    logLine("ONNX Runtime configured to use bundled webpack wasm assets.");
  }
}

async function handleModelSelection(event) {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }

  state.modelFile = file;
  state.modelSource = "uploaded-file";
  state.modelLabel = file.name;
  modelMeta.textContent = `Current model: ${file.name}`;
  modelStatus.textContent = `AI model selected: ${file.name}\nClick Load AI model.`;
  runButton.disabled = true;
  logLine(`Selected local AI model file: ${file.name} (${formatBytes(file.size)})`);
}

function handleModelSelectorChange(event) {
  const selectedValue = event.target.value;
  
  if (selectedValue === "upload") {
    modelUploadSection.classList.remove("hidden");
  } else {
    modelUploadSection.classList.add("hidden");
    modelInput.value = "";
    useBundledModel();
  }
}

function useBundledModel() {
  state.modelFile = null;
  state.modelSource = DEFAULT_MODEL_PATH;
  state.modelLabel = "Bundled AI model";
  modelInput.value = "";
  modelMeta.textContent = "Current model: bundled AI model";
  modelStatus.textContent = "Bundled AI model selected. Click Load AI model.";
  runButton.disabled = !state.imageBitmap;
  logLine(`Switched to bundled AI model: ${DEFAULT_MODEL_PATH}`);
}

async function loadSelectedModel() {
  loadModelButton.disabled = true;
  runButton.disabled = true;
  viewerHint.textContent = "Loading AI model...";

  try {
    logLine(`Starting AI model load: ${state.modelFile ? state.modelFile.name : DEFAULT_MODEL_PATH}`);
    const session = await createModelSession();
    state.session = session;
    describeLoadedModel(session);
    viewerHint.textContent = state.imageBitmap
      ? "AI model loaded. Ready to run analysis."
      : "AI model loaded. Upload an image to run analysis.";
    runButton.disabled = !state.imageBitmap;
  } catch (error) {
    state.session = null;
    modelStatus.textContent = "AI model could not be loaded.";
    viewerHint.textContent = "AI model load failed.";
    logLine(`AI model load failed: ${error.message}`, true);
  } finally {
    loadModelButton.disabled = false;
  }
}

async function createModelSession() {
  if (state.modelFile) {
    logLine("Reading uploaded AI model into memory...");
    const buffer = await state.modelFile.arrayBuffer();
    logLine(`Creating session from uploaded model bytes (${formatBytes(buffer.byteLength)})...`);
    return ort.InferenceSession.create(buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  logLine(`Fetching bundled model from ${DEFAULT_MODEL_PATH}...`);
  return ort.InferenceSession.create(DEFAULT_MODEL_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

function describeLoadedModel(session) {
  const [inputName] = session.inputNames;
  const dims = session.inputMetadata[inputName]?.dimensions || [];
  const rows = Number(dims[2]) || DEFAULT_INPUT_SIZE.rows;
  const cols = Number(dims[3]) || DEFAULT_INPUT_SIZE.cols;
  const sourceLabel = state.modelFile ? state.modelFile.name : DEFAULT_MODEL_PATH;

  modelStatus.textContent = [
    `AI model ready: ${sourceLabel}`,
    `Analysis input size: ${rows} x ${cols}`,
    `Outputs available: ${session.outputNames.length}`,
  ].join("\n");

  logLine(`AI model ready: ${sourceLabel}`);
  logLine(`Input tensor: ${inputName}`);
  logLine(`Expected shape: 1 x 3 x ${rows} x ${cols}`);
  logLine(`Outputs: ${session.outputNames.join(", ")}`);
}

async function handleImageSelection(event) {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }

  if (!["image/png", "image/jpeg"].includes(file.type)) {
    viewerHint.textContent = "Unsupported file type. Use PNG or JPEG.";
    logLine(`Rejected image file type: ${file.type || "unknown"}`, true);
    return;
  }

  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
  }

  state.imageBitmap?.close?.();
  state.imageName = file.name;
  state.imageUrl = URL.createObjectURL(file);
  state.imageBitmap = await createImageBitmap(file);

  state.overlayCanvas = null;
  drawBitmapToCanvas(state.imageBitmap, sourceCanvas);
  redrawViewer();
  viewerHint.textContent = state.session ? `Loaded ${file.name}. Ready to run analysis.` : `Loaded ${file.name}. Load an AI model to run analysis.`;
  imageMeta.textContent = `${file.name} | ${state.imageBitmap.width} x ${state.imageBitmap.height} px`;
  runButton.disabled = !state.session;
  setZoomControlsEnabled(true);
  resetZoom();
  logLine(`Loaded image: ${file.name} (${state.imageBitmap.width} x ${state.imageBitmap.height})`);
}

async function runInference() {
  if (!state.session || !state.imageBitmap) {
    logLine("Inference skipped: model or image missing.", true);
    return;
  }

  runButton.disabled = true;
  viewerHint.textContent = "Running analysis...";
  logLine("Preparing input tensor...");

  try {
    const [inputName] = state.session.inputNames;
    const inputShape = getInputShape(state.session, inputName);
    const tensor = imageBitmapToTensor(state.imageBitmap, inputShape.cols, inputShape.rows);
    logLine(`Running session with input ${inputName} shaped 1 x 3 x ${inputShape.rows} x ${inputShape.cols}...`);
    const results = await state.session.run({ [inputName]: tensor });
    logLine(`Inference completed. Returned outputs: ${Object.keys(results).join(", ")}`);
    const outputTensor = selectSegmentationOutput(results);
    const mask = tensorToMask(outputTensor);

    drawOverlay(mask, state.imageBitmap.width, state.imageBitmap.height);
    viewerHint.textContent = "Analysis complete. Overlay updated.";
    logLine(`Overlay drawn from output dims ${outputTensor.dims.join(" x ")}`);
  } catch (error) {
    viewerHint.textContent = "Analysis failed. See diagnostics for details.";
    logLine(`Inference failed: ${error.message}`, true);
  } finally {
    runButton.disabled = false;
  }
}

function getInputShape(session, inputName) {
  const dims = session.inputMetadata[inputName]?.dimensions || [];
  return {
    rows: Number(dims[2]) || DEFAULT_INPUT_SIZE.rows,
    cols: Number(dims[3]) || DEFAULT_INPUT_SIZE.cols,
  };
}

function imageBitmapToTensor(imageBitmap, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(imageBitmap, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  const floatData = new Float32Array(width * height * 3);
  const channelSize = width * height;

  for (let index = 0; index < width * height; index += 1) {
    const pixelOffset = index * 4;
    floatData[index] = data[pixelOffset] / 255;
    floatData[channelSize + index] = data[pixelOffset + 1] / 255;
    floatData[channelSize * 2 + index] = data[pixelOffset + 2] / 255;
  }

  return new ort.Tensor("float32", floatData, [1, 3, height, width]);
}

function selectSegmentationOutput(results) {
  if (results.output1) {
    return results.output1;
  }

  return Object.values(results)
    .filter((tensor) => Array.isArray(tensor.dims) && tensor.dims.length >= 4)
    .sort((left, right) => {
      const leftArea = (left.dims.at(-1) || 1) * (left.dims.at(-2) || 1);
      const rightArea = (right.dims.at(-1) || 1) * (right.dims.at(-2) || 1);
      return rightArea - leftArea;
    })[0] || Object.values(results)[0];
}

function tensorToMask(tensor) {
  const width = tensor.dims.at(-1);
  const height = tensor.dims.at(-2);
  const data = tensor.data;
  const pixels = new Uint8ClampedArray(width * height * 4);
  const channelStride = width * height;

  for (let index = 0; index < width * height; index += 1) {
    const raw = data[index] ?? data[index % channelStride] ?? 0;
    const score = sigmoid(raw);
    const alpha = score > 0.5 ? Math.round(score * 180) : 0;
    const pixelOffset = index * 4;
    pixels[pixelOffset] = 239;
    pixels[pixelOffset + 1] = 68;
    pixels[pixelOffset + 2] = 68;
    pixels[pixelOffset + 3] = alpha;
  }

  return new ImageData(pixels, width, height);
}

function drawOverlay(maskImageData, targetWidth, targetHeight) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = maskImageData.width;
  tempCanvas.height = maskImageData.height;
  tempCanvas.getContext("2d").putImageData(maskImageData, 0, 0);

  state.overlayCanvas = document.createElement("canvas");
  state.overlayCanvas.width = targetWidth;
  state.overlayCanvas.height = targetHeight;
  const overlayContext = state.overlayCanvas.getContext("2d");
  overlayContext.clearRect(0, 0, targetWidth, targetHeight);
  overlayContext.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
  redrawViewer();
}

function drawBitmapToCanvas(imageBitmap, canvas) {
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(imageBitmap, 0, 0);
}

function clearCanvas(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").clearRect(0, 0, width, height);
}

function redrawViewer() {
  if (!state.imageBitmap) {
    return;
  }

  viewerCanvas.width = sourceCanvas.width;
  viewerCanvas.height = sourceCanvas.height;
  const context = viewerCanvas.getContext("2d");
  context.clearRect(0, 0, viewerCanvas.width, viewerCanvas.height);
  context.drawImage(sourceCanvas, 0, 0);

  if (state.overlayCanvas && state.overlayVisible) {
    context.save();
    context.globalAlpha = Number(overlayOpacity.value) / 100;
    context.drawImage(state.overlayCanvas, 0, 0);
    context.restore();
  }

  renderMinimap();
}

function toggleDiagnostics() {
  state.diagnosticsVisible = !state.diagnosticsVisible;
  diagnosticsPanel.classList.toggle("hidden", !state.diagnosticsVisible);
  diagnosticsPanel.setAttribute("aria-hidden", state.diagnosticsVisible ? "false" : "true");
  diagnosticsToggleButton.textContent = state.diagnosticsVisible ? "Hide diagnostics" : "Show diagnostics";
}

function toggleControls() {
  state.controlsCollapsed = !state.controlsCollapsed;
  workspaceLayout.classList.toggle("controls-collapsed", state.controlsCollapsed);
  toolbarPanel.classList.toggle("collapsed", state.controlsCollapsed);
  toggleControlsButton.textContent = state.controlsCollapsed ? "Expand" : "Collapse";
}

function changeZoom(delta, anchorX, anchorY) {
  if (!state.imageBitmap) {
    return;
  }

  const stageRect = viewerStage.getBoundingClientRect();
  const focusX = anchorX ?? stageRect.width / 2;
  const focusY = anchorY ?? stageRect.height / 2;
  const previousZoom = state.zoom;
  const nextZoom = clamp(previousZoom + delta, 0.4, 8);

  if (nextZoom === previousZoom) {
    return;
  }

  const worldX = (focusX - state.panX) / previousZoom;
  const worldY = (focusY - state.panY) / previousZoom;

  state.zoom = nextZoom;
  state.panX = focusX - worldX * nextZoom;
  state.panY = focusY - worldY * nextZoom;
  applyZoomTransform();
}

function resetZoom() {
  if (!state.imageBitmap) {
    return;
  }

  const stageWidth = viewerStage.clientWidth;
  const stageHeight = viewerStage.clientHeight;
  const fitX = stageWidth / sourceCanvas.width;
  const fitY = stageHeight / sourceCanvas.height;
  state.zoom = clamp(Math.min(fitX, fitY), 0.4, 4);

  const renderedWidth = sourceCanvas.width * state.zoom;
  const renderedHeight = sourceCanvas.height * state.zoom;
  state.panX = (stageWidth - renderedWidth) / 2;
  state.panY = (stageHeight - renderedHeight) / 2;

  applyZoomTransform();
}

function handleZoomWheel(event) {
  if (!state.imageBitmap) {
    return;
  }

  event.preventDefault();
  const rect = viewerStage.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;
  changeZoom(event.deltaY < 0 ? 0.16 : -0.16, localX, localY);
}

function startPan(event) {
  if (!state.imageBitmap || event.button !== 0) {
    return;
  }

  state.isDragging = true;
  state.activePointerId = event.pointerId;
  state.dragStartX = event.clientX - state.panX;
  state.dragStartY = event.clientY - state.panY;
  viewerStage.classList.add("dragging");
  viewerStage.setPointerCapture(event.pointerId);
}

function movePan(event) {
  if (!state.isDragging || state.activePointerId !== event.pointerId) {
    return;
  }

  state.panX = event.clientX - state.dragStartX;
  state.panY = event.clientY - state.dragStartY;
  applyZoomTransform();
}

function endPan(event) {
  if (event && state.activePointerId !== null && state.activePointerId !== event.pointerId) {
    return;
  }

  if (event && viewerStage.hasPointerCapture(event.pointerId)) {
    viewerStage.releasePointerCapture(event.pointerId);
  }

  state.isDragging = false;
  state.activePointerId = null;
  viewerStage.classList.remove("dragging");
}

function applyZoomTransform() {
  viewerCanvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
  renderMinimap();
}

function setZoomControlsEnabled(enabled) {
  zoomInButton.disabled = !enabled;
  zoomOutButton.disabled = !enabled;
  zoomResetButton.disabled = !enabled;
  miniMapWrap.classList.toggle("hidden", !enabled);
  miniMapWrap.setAttribute("aria-hidden", enabled ? "false" : "true");
  if (!enabled) {
    zoomLevel.textContent = "100%";
  }
}

function renderMinimap() {
  if (!state.imageBitmap) {
    return;
  }

  const ctx = miniMapCanvas.getContext("2d");
  const mapWidth = miniMapCanvas.width;
  const mapHeight = miniMapCanvas.height;

  ctx.clearRect(0, 0, mapWidth, mapHeight);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, mapWidth, mapHeight);

  const scale = Math.min(mapWidth / sourceCanvas.width, mapHeight / sourceCanvas.height);
  const drawWidth = sourceCanvas.width * scale;
  const drawHeight = sourceCanvas.height * scale;
  const offsetX = (mapWidth - drawWidth) / 2;
  const offsetY = (mapHeight - drawHeight) / 2;

  // Draw original image from sourceCanvas into minimap
  ctx.drawImage(sourceCanvas, offsetX, offsetY, drawWidth, drawHeight);

  // Calculate which part of the image is visible in the main viewport
  // Position in world coordinates
  const worldLeft = -state.panX / state.zoom;
  const worldTop = -state.panY / state.zoom;
  const worldWidth = viewerStage.clientWidth / state.zoom;
  const worldHeight = viewerStage.clientHeight / state.zoom;

  // Convert to minimap coordinates
  const viewportLeft = worldLeft * scale + offsetX;
  const viewportTop = worldTop * scale + offsetY;
  const viewportWidth = worldWidth * scale;
  const viewportHeight = worldHeight * scale;

  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2;
  ctx.strokeRect(viewportLeft, viewportTop, viewportWidth, viewportHeight);
}

function logLine(message, isError = false) {
  const timestamp = new Date().toLocaleTimeString();
  const line = `[${timestamp}] ${isError ? "ERROR: " : ""}${message}`;
  if (!logOutput.textContent || logOutput.textContent === "Diagnostics are ready." || logOutput.textContent === "Diagnostics cleared.") {
    logOutput.textContent = line;
  } else {
    logOutput.textContent += `\n${line}`;
  }
  logOutput.scrollTop = logOutput.scrollHeight;
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
