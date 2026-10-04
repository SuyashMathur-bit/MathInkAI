# MathInk.ai — AI-Powered Mathematical Smart Board

MathInk.ai is an end-to-end mathematical handwriting recognition and symbolic reasoning system. It enables users to naturally write mathematical expressions on a digital smart board using a mouse, stylus, or touchscreen, and automatically recognizes the handwriting using a CNN + BiLSTM + CTC deep learning pipeline and solves the equation using symbolic mathematics.

---

## Conceptual Pipeline

```
CROHME Processed Dataset
        ↓
LaTeX Tokenization (Regex)
        ↓
Vocabulary Creation (518 tokens, PAD=0, UNK=1, BLANK=517)
        ↓
Stroke → [x, y, pen] Sequence Representation
        ↓
Train / Validation Split (80 / 20)
        ↓
Padding + Length Preparation
        ↓
CTC Preparation (Conv1D + MaxPool1D downsampling // 2)
        ↓
Model: Conv1D(64, k=5) → MaxPool1D(2) → BiLSTM(128) → BiLSTM(128) → Dense(518, Softmax)
        ↓
Resumable CTC Training with Per-Epoch Checkpoints (1 to 50 Epochs)
        ↓
Inference Model
        ↓
Greedy CTC Decoding (deduplication & blank suppression)
        ↓
Predicted Tokens
        ↓
Tokens → LaTeX (`"".join(tokens)`)
        ↓
LaTeX → SymPy (`latex2sympy2` + `sympy`)
        ↓
solve_latex() / step_by_step_solution()
        ↓
Mathematical Solution (Roots, Step Cards, Educational Explanations)
```

---

## Project Structure

```
Math.ai/
├── MathInk_AI_Pipeline.ipynb           # Complete 85-cell Jupyter Notebook (for Kaggle / Colab)
├── mathink_ai_pipeline.py              # Standalone Python notebook script (Epoch 71)
├── mathink_ai.py                       # Modular MathInkRecognizer API (Online & Image)
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules for dataset & checkpoints
├── dataset/
│   └── processed_crohme.pkl            # CROHME dataset (21,265 samples)
├── model/
│   ├── mathinkai_epoch_71.weights.h5   # Epoch 71 model weights
│   ├── best_mathink_model.weights.h5   # Best validation-performing weights
│   ├── vocabulary.pkl                  # 518 vocabulary tokens
│   ├── token_to_id.pkl                 # Token to ID mapping
│   ├── id_to_token.pkl                 # ID to Token mapping
│   ├── model_config.json               # Architecture hyperparameters
│   └── build_vocab.py                  # Vocabulary extraction script
├── training/
│   ├── train.py                        # Resumable training loop with checkpointing
│   ├── evaluation.py                   # 300-sample validation evaluation
│   ├── evaluation_report.json          # Metrics report
│   └── checkpoints/                    # Epoch checkpoints (up to Epoch 80)
│       ├── checkpoint_state.json       # Resumable training state
│       ├── mathink_checkpoint_epoch_71.weights.h5
│       └── best_mathink_model.weights.h5
├── inference/
│   ├── preprocessing.py                # Stroke normalization & tokenization
│   ├── model_loader.py                 # Architecture builder & inference pipeline
│   ├── decoding.py                     # Greedy CTC decode
│   ├── latex.py                        # Tokens to LaTeX & expression validation
│   └── solver.py                       # Step-by-step symbolic solver
├── backend/
│   └── app.py                          # Flask REST API server
├── frontend/
│   ├── index.html                      # Digital Smart Board interface
│   ├── style.css                       # Responsive dark slate design
│   └── app.js                          # Real-time stroke capture & KaTeX rendering
├── run_app.bat                         # One-click Windows startup script
└── README.md
```

---

## Resumable Training & Checkpoints

The training script automatically detects previously completed epochs and continues from the next epoch until reaching a total of 50 epochs:

```powershell
# Resume training automatically to 50 epochs
py -3.11 training\train.py --epochs 50

# Optional subset training for rapid iteration
py -3.11 training\train.py --epochs 50 --max-samples 500

# Force restart from epoch 1
py -3.11 training\train.py --epochs 50 --force-restart
```

### Checkpoint State File (`training/checkpoints/checkpoint_state.json`)
Saves:
- `completed_epoch`
- `best_epoch`
- `best_val_loss`
- `best_checkpoint_path`
- Per-epoch history (`train_loss`, `val_loss`, `is_best`, `time_seconds`)

---

## Validation Evaluation

The model was evaluated on 300 validation samples matching the project methodology:

```powershell
py -3.11 training\evaluation.py --samples 300
```

### Benchmark Progression

| Milestone | Token Precision | Token Recall | Token F1 | Exact Match | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Epoch 10** | - | - | - | 9.00% | Initial baseline |
| **Epoch 20** | - | - | - | 37.67% | Intermediate checkpoint |
| **Epoch 50** | 49.75% | 9.37% | 15.77% | 0.00% | Validation Loss: 60.1154 |
| **Epoch 71 (Best)** | **81.60%** | **47.82%** | **60.30%** | **3.67%** | **Validation Loss: 24.7543** (Active Best Model) |

--- | :--- | :--- | :--- | :--- |
| **Epoch 10** | 36.43% | 9.00% | — | Baseline early checkpoint |
| **Epoch 20** | 59.98% | 37.67% | 83.67% | Intermediate notebook checkpoint |
| **Epoch 50** | Evaluated | Evaluated | Evaluated | `best_mathink_model` selected at Epoch 48 (val loss: 60.1154) |

---

## Running the Smart Board Application

### 1. Launch the Server
Double-click `run_app.bat` or run:
```powershell
py -3.11 backend\app.py
```
Server runs at `http://localhost:5000`.

### 2. Open in Browser
Open `http://localhost:5000` in Google Chrome, Edge, or on any touchscreen/tablet.

### 3. Features
- **Canvas**: Natural writing with mouse, stylus (Surface Pen / Apple Pencil), or touchscreen.
- **Stroke Capture**: Captures exact continuous sequences of `[x, y, pen]`.
- **Tools**: Pen, Eraser, Undo, Redo, Clear Board.
- **Try Example**: Dropdown to load sample equations from the CROHME validation set.
- **Solve Button**: Prominent `✨ Solve` button with animated progressive feedback.
- **LaTeX Display**: Beautiful KaTeX rendered mathematical equations.
- **Mathematical Solution**: Real SymPy symbolic solver displaying roots, factorizations, and step cards.
- **Developer Mode**: Real-time telemetry inspector (strokes, points, input shape, inference latency, raw token IDs, LaTeX string).
