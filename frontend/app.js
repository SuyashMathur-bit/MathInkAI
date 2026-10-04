/**
 * MathInk.ai — AI-Powered Mathematical Smart Board
 * Real-time stroke capturing, rendering, CTC recognition & mathematical solver.
 */

// Global State
const state = {
  activeTool: 'pen', // 'pen' | 'eraser'
  strokes: [], // Array of strokes, each is array of [x, y]
  redoStack: [],
  isDrawing: false,
  currentStroke: [],
  samples: [],
  isDevMode: false,
  isSolving: false
};

// DOM Elements
const canvas = document.getElementById('smartBoardCanvas');
const ctx = canvas.getContext('2d');
const canvasContainer = document.getElementById('canvasContainer');
const canvasWatermark = document.getElementById('canvasWatermark');

const penToolBtn = document.getElementById('penToolBtn');
const eraserToolBtn = document.getElementById('eraserToolBtn');
const undoBtn = document.getElementById('undoBtn');
const redoBtn = document.getElementById('redoBtn');
const clearBtn = document.getElementById('clearBtn');
const uploadImgBtn = document.getElementById('uploadImgBtn');
const imageFileInput = document.getElementById('imageFileInput');
const dropOverlay = document.getElementById('dropOverlay');
const solveBtn = document.getElementById('solveBtn');
const exampleSelector = document.getElementById('exampleSelector');

const strokeCounter = document.getElementById('strokeCounter');
const pointCounter = document.getElementById('pointCounter');
const modelStatusBadge = document.getElementById('modelStatusBadge');
const modelStatusText = document.getElementById('modelStatusText');
const devModeToggleBtn = document.getElementById('devModeToggleBtn');

const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const devPanel = document.getElementById('devPanel');

const equationRenderArea = document.getElementById('equationRenderArea');
const solutionRenderArea = document.getElementById('solutionRenderArea');
const expressionTypeTag = document.getElementById('expressionTypeTag');
const solutionStatusTag = document.getElementById('solutionStatusTag');
const stepsContainer = document.getElementById('stepsContainer');
const stepsList = document.getElementById('stepsList');

const editLatexToggleBtn = document.getElementById('editLatexToggleBtn');
const latexEditorDrawer = document.getElementById('latexEditorDrawer');
const latexInput = document.getElementById('latexInput');
const reSolveBtn = document.getElementById('reSolveBtn');

// Pipeline Steps
const stepElements = {
  read: document.getElementById('stepRead'),
  recognize: document.getElementById('stepRecognize'),
  latex: document.getElementById('stepLatex'),
  solve: document.getElementById('stepSolve'),
  ready: document.getElementById('stepReady')
};

// Dev Elements
const devStrokes = document.getElementById('devStrokes');
const devPoints = document.getElementById('devPoints');
const devShape = document.getElementById('devShape');
const devLatency = document.getElementById('devLatency');
const devCheckpoint = document.getElementById('devCheckpoint');
const devExprType = document.getElementById('devExprType');
const devTokens = document.getElementById('devTokens');
const devRawLatex = document.getElementById('devRawLatex');

// Setup High-DPI Canvas
function resizeCanvas() {
  const rect = canvasContainer.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  redrawAllStrokes();
}

window.addEventListener('resize', resizeCanvas);

// Canvas Coordinates helper
function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
  const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);

  return [
    clientX - rect.left,
    clientY - rect.top
  ];
}

// Drawing Logic
function startDrawing(e) {
  if (state.isSolving) return;
  e.preventDefault();
  state.isDrawing = true;

  const [x, y] = getCanvasCoords(e);

  if (state.activeTool === 'eraser') {
    eraseNearPoint(x, y);
    return;
  }

  state.currentStroke = [[x, y]];
  state.redoStack = []; // New stroke clears redo

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.strokeStyle = '#38bdf8'; // Cyan math chalk
  ctx.lineWidth = 3.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  canvasWatermark.classList.add('hidden');
}

function continueDrawing(e) {
  if (!state.isDrawing || state.isSolving) return;
  e.preventDefault();

  const [x, y] = getCanvasCoords(e);

  if (state.activeTool === 'eraser') {
    eraseNearPoint(x, y);
    return;
  }

  state.currentStroke.push([x, y]);

  // Smooth rendering using midpoint quadratic bezier
  const pts = state.currentStroke;
  const len = pts.length;

  if (len >= 3) {
    const xc = (pts[len - 2][0] + pts[len - 1][0]) / 2;
    const yc = (pts[len - 2][1] + pts[len - 1][1]) / 2;
    ctx.quadraticCurveTo(pts[len - 2][0], pts[len - 2][1], xc, yc);
  } else {
    ctx.lineTo(x, y);
  }

  ctx.stroke();
  updateMetricsDisplay();
}

function stopDrawing(e) {
  if (!state.isDrawing) return;
  state.isDrawing = false;

  if (state.activeTool === 'pen' && state.currentStroke.length > 0) {
    state.strokes.push([...state.currentStroke]);
    state.currentStroke = [];
  }

  redrawAllStrokes();
  updateMetricsDisplay();
  updateButtonStates();
}

// Stroke Eraser
function eraseNearPoint(x, y, radius = 22) {
  let changed = false;
  state.strokes = state.strokes.filter(stroke => {
    // Keep stroke if no point is within radius
    const hit = stroke.some(pt => {
      const dx = pt[0] - x;
      const dy = pt[1] - y;
      return (dx * dx + dy * dy) <= (radius * radius);
    });
    if (hit) changed = true;
    return !hit;
  });

  if (changed) {
    redrawAllStrokes();
    updateMetricsDisplay();
    updateButtonStates();
  }
}

// Redraw all strokes on canvas
function redrawAllStrokes() {
  const rect = canvasContainer.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (state.strokes.length === 0 && state.currentStroke.length === 0) {
    canvasWatermark.classList.remove('hidden');
    return;
  }

  canvasWatermark.classList.add('hidden');

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 3.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const allToDraw = [...state.strokes];
  if (state.currentStroke.length > 0) {
    allToDraw.push(state.currentStroke);
  }

  allToDraw.forEach(stroke => {
    if (stroke.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(stroke[0][0], stroke[0][1]);

    if (stroke.length === 1) {
      ctx.arc(stroke[0][0], stroke[0][1], 1.8, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      return;
    }

    for (let i = 1; i < stroke.length - 1; i++) {
      const xc = (stroke[i][0] + stroke[i + 1][0]) / 2;
      const yc = (stroke[i][1] + stroke[i + 1][1]) / 2;
      ctx.quadraticCurveTo(stroke[i][0], stroke[i][1], xc, yc);
    }
    ctx.lineTo(stroke[stroke.length - 1][0], stroke[stroke.length - 1][1]);
    ctx.stroke();
  });
}

// Undo & Redo
function undo() {
  if (state.strokes.length === 0) return;
  const popped = state.strokes.pop();
  state.redoStack.push(popped);
  redrawAllStrokes();
  updateMetricsDisplay();
  updateButtonStates();
}

function redo() {
  if (state.redoStack.length === 0) return;
  const restored = state.redoStack.pop();
  state.strokes.push(restored);
  redrawAllStrokes();
  updateMetricsDisplay();
  updateButtonStates();
}

function clearBoard() {
  state.strokes = [];
  state.redoStack = [];
  state.currentStroke = [];
  redrawAllStrokes();
  updateMetricsDisplay();
  updateButtonStates();
  resultsSection.style.display = 'none';
  processingSection.style.display = 'none';
}

function updateMetricsDisplay() {
  const strokeCount = state.strokes.length + (state.currentStroke.length > 0 ? 1 : 0);
  let ptCount = state.strokes.reduce((acc, s) => acc + s.length, 0) + state.currentStroke.length;

  strokeCounter.textContent = `${strokeCount} stroke${strokeCount === 1 ? '' : 's'}`;
  pointCounter.textContent = `${ptCount} point${ptCount === 1 ? '' : 's'}`;
}

function updateButtonStates() {
  undoBtn.disabled = state.strokes.length === 0;
  redoBtn.disabled = state.redoStack.length === 0;
}

// Tool Switching
penToolBtn.addEventListener('click', () => {
  state.activeTool = 'pen';
  penToolBtn.classList.add('active');
  eraserToolBtn.classList.remove('active');
  canvas.style.cursor = 'crosshair';
});

eraserToolBtn.addEventListener('click', () => {
  state.activeTool = 'eraser';
  eraserToolBtn.classList.add('active');
  penToolBtn.classList.remove('active');
  canvas.style.cursor = 'pointer';
});

undoBtn.addEventListener('click', undo);
redoBtn.addEventListener('click', redo);
clearBtn.addEventListener('click', clearBoard);

// Pointer Events (supports stylus, finger, mouse seamlessly)
canvas.addEventListener('pointerdown', startDrawing);
canvas.addEventListener('pointermove', continueDrawing);
canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointercancel', stopDrawing);
canvas.addEventListener('pointerleave', stopDrawing);

// Keyboard shortcuts (Ctrl+Z, Ctrl+Y)
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    if (e.shiftKey) redo();
    else undo();
  } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
    redo();
  }
});

// Load Quick Examples onto Canvas
function loadSampleStrokes(sample) {
  if (!sample || !sample.strokes) return;

  const rect = canvasContainer.getBoundingClientRect();
  const targetW = rect.width * 0.75;
  const targetH = rect.height * 0.55;

  // Find sample bounding box
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  sample.strokes.forEach(s => {
    s.forEach(pt => {
      minX = Math.min(minX, pt[0]);
      maxX = Math.max(maxX, pt[0]);
      minY = Math.min(minY, pt[1]);
      maxY = Math.max(maxY, pt[1]);
    });
  });

  const rawW = maxX - minX || 1;
  const rawH = maxY - minY || 1;
  const scale = Math.min(targetW / rawW, targetH / rawH);

  const offsetX = (rect.width - rawW * scale) / 2 - minX * scale;
  const offsetY = (rect.height - rawH * scale) / 2 - minY * scale;

  state.strokes = sample.strokes.map(s => {
    return s.map(pt => [
      pt[0] * scale + offsetX,
      pt[1] * scale + offsetY
    ]);
  });

  state.redoStack = [];
  redrawAllStrokes();
  updateMetricsDisplay();
  updateButtonStates();

  // Trigger solve automatically for quick example review
  executeSolve();
}

// Animated Step Progression Helper
async function animatePipelineStep(stepKey, delayMs = 180) {
  Object.values(stepElements).forEach(el => el.classList.remove('active'));
  stepElements[stepKey].classList.add('active');
  await new Promise(r => setTimeout(r, delayMs));
  stepElements[stepKey].classList.remove('active');
  stepElements[stepKey].classList.add('completed');
}

function resetPipelineSteps() {
  Object.values(stepElements).forEach(el => {
    el.classList.remove('active', 'completed');
  });
}

// Solve Execution
async function executeSolve() {
  if (state.strokes.length === 0) {
    alert("No handwriting detected. Please write a mathematical equation on the smart board first!");
    return;
  }

  state.isSolving = true;
  solveBtn.disabled = true;
  processingSection.style.display = 'block';
  resultsSection.style.display = 'none';
  resetPipelineSteps();

  try {
    await animatePipelineStep('read', 200);

    const payload = {
      strokes: state.strokes
    };

    const fetchPromise = fetch('/api/recognize_and_solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    await animatePipelineStep('recognize', 250);
    await animatePipelineStep('latex', 200);

    const response = await fetchPromise;
    const result = await response.json();

    await animatePipelineStep('solve', 220);
    await animatePipelineStep('ready', 150);

    displayResults(result);

  } catch (err) {
    console.error("Inference Error:", err);
    alert("Recognition failed: Unable to connect to backend server.");
  } finally {
    state.isSolving = false;
    solveBtn.disabled = false;
    setTimeout(() => {
      processingSection.style.display = 'none';
    }, 400);
  }
}

solveBtn.addEventListener('click', executeSolve);

// Image Upload & Extraction Processing
function processImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    alert("Please select or drop an image file (PNG, JPG, WEBP).");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadAndSolveImage(e.target.result);
  };
  reader.readAsDataURL(file);
}

async function uploadAndSolveImage(base64Data) {
  state.isSolving = true;
  solveBtn.disabled = true;
  processingSection.style.display = 'block';
  resultsSection.style.display = 'none';
  resetPipelineSteps();

  try {
    await animatePipelineStep('read', 250);

    const fetchPromise = fetch('/api/upload_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: base64Data })
    });

    await animatePipelineStep('recognize', 280);
    await animatePipelineStep('latex', 200);

    const response = await fetchPromise;
    const result = await response.json();

    if (!result.success) {
      throw new Error(result.error || "Failed to process image");
    }

    // Load extracted strokes onto the smart board
    if (result.extracted_strokes && result.extracted_strokes.length > 0) {
      state.strokes = result.extracted_strokes;
      state.redoStack = [];
      redrawAllStrokes();
      updateMetricsDisplay();
      updateButtonStates();
    }

    await animatePipelineStep('solve', 220);
    await animatePipelineStep('ready', 150);

    displayResults(result);

  } catch (err) {
    console.error("Image Processing Error:", err);
    alert("Image recognition failed: " + (err.message || "Unknown error"));
  } finally {
    state.isSolving = false;
    solveBtn.disabled = false;
    setTimeout(() => {
      processingSection.style.display = 'none';
    }, 400);
  }
}

// Upload Button & File Input Event Listeners
if (uploadImgBtn) {
  uploadImgBtn.addEventListener('click', () => {
    if (imageFileInput) imageFileInput.click();
  });
}

if (imageFileInput) {
  imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processImageFile(e.target.files[0]);
      imageFileInput.value = '';
    }
  });
}

// Drag & Drop Listeners on Canvas Container
if (canvasContainer && dropOverlay) {
  ['dragenter', 'dragover'].forEach(evtName => {
    canvasContainer.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      canvasContainer.classList.add('drag-active');
      dropOverlay.classList.add('visible');
    });
  });

  ['dragleave', 'dragend'].forEach(evtName => {
    canvasContainer.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!canvasContainer.contains(e.relatedTarget)) {
        canvasContainer.classList.remove('drag-active');
        dropOverlay.classList.remove('visible');
      }
    });
  });

  canvasContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    canvasContainer.classList.remove('drag-active');
    dropOverlay.classList.remove('visible');

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processImageFile(e.dataTransfer.files[0]);
    }
  });
}

// Clipboard Paste (Ctrl+V) for screenshots
window.addEventListener('paste', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData))?.items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {
    if (items[i].type && items[i].type.startsWith('image/')) {
      const file = items[i].getAsFile();
      if (file) {
        processImageFile(file);
        break;
      }
    }
  }
});

// Display Results
function displayResults(result) {
  resultsSection.style.display = 'grid';

  if (!result.success) {
    equationRenderArea.innerHTML = `<span style="color: var(--accent-rose); font-size: 16px;">${result.error || "Unable to recognize expression"}</span>`;
    solutionRenderArea.innerHTML = `<span style="color: var(--text-dim); font-size: 16px;">No solution available</span>`;
    stepsContainer.style.display = 'none';
    return;
  }

  const rawLatex = result.latex || "";
  latexInput.value = rawLatex;

  expressionTypeTag.textContent = result.expression_type || "Expression";
  solutionStatusTag.textContent = result.status || "Ready";

  // Render LaTeX Equation using KaTeX
  try {
    if (window.katex && rawLatex) {
      katex.render(rawLatex, equationRenderArea, {
        throwOnError: false,
        displayMode: true
      });
    } else {
      equationRenderArea.textContent = rawLatex || "(Empty)";
    }
  } catch (e) {
    equationRenderArea.textContent = rawLatex;
  }

  // Render Mathematical Solution
  const solutionText = result.solution || "None";
  try {
    if (window.katex && solutionText) {
      katex.render(solutionText, solutionRenderArea, {
        throwOnError: false,
        displayMode: true
      });
    } else {
      solutionRenderArea.textContent = solutionText;
    }
  } catch (e) {
    solutionRenderArea.textContent = solutionText;
  }

  // Render Step-by-Step Breakdown Cards
  const steps = result.steps || [];
  if (steps.length > 0) {
    stepsContainer.style.display = 'flex';
    stepsList.innerHTML = '';
    steps.forEach((step, idx) => {
      const card = document.createElement('div');
      card.className = 'step-card';

      // If step contains LaTeX formulas $$, render with KaTeX
      card.innerHTML = renderStepWithKatex(step);
      stepsList.appendChild(card);
    });
  } else {
    stepsContainer.style.display = 'none';
  }

  // Populate Developer Inspector
  devStrokes.textContent = result.strokes_count || state.strokes.length;
  devPoints.textContent = result.points_count || "--";
  devShape.textContent = result.input_shape ? `(${result.input_shape.join(', ')})` : "--";
  devLatency.textContent = `${result.inference_time_ms || '--'} ms (model: ${result.model_time_ms || '--'} ms)`;
  devCheckpoint.textContent = result.checkpoint_info ? result.checkpoint_info.filename : "Loaded";
  devExprType.textContent = result.expression_type || "--";
  devTokens.textContent = JSON.stringify(result.tokens || [], null, 2);
  devRawLatex.textContent = rawLatex;
}

// Render steps with KaTeX
function renderStepWithKatex(text) {
  if (!text) return "";
  // Look for $$...$$
  return text.replace(/\$\$(.+?)\$\$/g, (match, formula) => {
    try {
      return katex.renderToString(formula, { throwOnError: false });
    } catch (e) {
      return formula;
    }
  });
}

// LaTeX Editor Toggle & Re-solve
editLatexToggleBtn.addEventListener('click', () => {
  const isHidden = latexEditorDrawer.style.display === 'none';
  latexEditorDrawer.style.display = isHidden ? 'block' : 'none';
  if (isHidden) latexInput.focus();
});

reSolveBtn.addEventListener('click', async () => {
  const editedLatex = latexInput.value.trim();
  if (!editedLatex) return;

  reSolveBtn.disabled = true;
  try {
    const res = await fetch('/api/solve_latex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latex: editedLatex })
    });
    const data = await res.json();
    displayResults({
      success: true,
      latex: editedLatex,
      expression_type: data.expression_type,
      status: data.status,
      solution: data.solution,
      steps: data.steps,
      strokes_count: state.strokes.length,
      points_count: state.strokes.reduce((a, s) => a + s.length, 0),
      inference_time_ms: data.solve_time_ms
    });
  } catch (e) {
    alert("Could not re-solve expression: " + e.message);
  } finally {
    reSolveBtn.disabled = false;
  }
});

// Dev Mode Toggle
devModeToggleBtn.addEventListener('click', () => {
  state.isDevMode = !state.isDevMode;
  devPanel.style.display = state.isDevMode ? 'flex' : 'none';
  devModeToggleBtn.classList.toggle('active', state.isDevMode);
});

// Fetch Server Status & Preloaded Samples
async function initApp() {
  resizeCanvas();

  try {
    const res = await fetch('/api/status');
    const statusData = await res.json();
    if (statusData.checkpoint_loaded) {
      modelStatusText.textContent = `Model Ready (${statusData.checkpoint})`;
    } else {
      modelStatusText.textContent = `Base Architecture Ready`;
    }
  } catch (err) {
    modelStatusText.textContent = `Offline`;
    modelStatusBadge.querySelector('.status-dot').style.background = 'var(--accent-rose)';
  }

  try {
    const sRes = await fetch('/api/samples');
    const sData = await sRes.json();
    if (sData.success && sData.samples) {
      state.samples = sData.samples;
      sData.samples.forEach(sample => {
        const opt = document.createElement('option');
        opt.value = sample.id;
        opt.textContent = `${sample.title} (${sample.stroke_count} strokes)`;
        exampleSelector.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load samples:", err);
  }
}

exampleSelector.addEventListener('change', (e) => {
  const selectedId = e.target.value;
  if (!selectedId) return;
  const sample = state.samples.find(s => s.id === selectedId);
  if (sample) {
    loadSampleStrokes(sample);
  }
});

// Start app
initApp();
