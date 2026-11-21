# MathInk.ai — Deep Learning Handwritten Math to LaTeX Recognizer

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CROHME%20(21k)-green.svg)](https://www.kaggle.com/datasets/suyashma/math-expression1)

MathInk.ai is an end-to-end deep learning system that recognizes online handwritten mathematical expressions and converts them into standardized **LaTeX** code using a hybrid **Conv1D + Bidirectional LSTM + CTC (Connectionist Temporal Classification)** architecture trained on the benchmark **CROHME** dataset (21,265 samples).

---

## 📌 Architecture & Conceptual Pipeline

```
Online Handwritten Stroke Data [[x1, y1], [x2, y2], ...]
                          │
                          ▼
        Stroke-to-Sequence Representation [x, y, pen_state]
                          │
                          ▼
           Regex Tokenization & Vocabulary (518 Tokens)
                          │
                          ▼
           Conv1D (64 filters, kernel=5) + MaxPool1D(2)
                          │
                          ▼
           2x Bidirectional LSTM (128 units each)
                          │
                          ▼
        Dense Projection (518 classes, Softmax Activation)
                          │
                          ▼
               CTC Loss & Training Optimization
                          │
                          ▼
        Greedy CTC Decoding (Blank Suppression & Deduplication)
                          │
                          ▼
        Predicted LaTeX Token Sequence (e.g. x^2 + 5x + 6 = 0)
```

---

## 🎯 Key Technical Highlights

* **Temporal Representation of Handwriting:** In online handwriting recognition, mathematical formulas are captured as continuous trajectories of coordinates with pen-down/pen-up states `[x, y, pen_state]` rather than static 2D images, preserving crucial writing order and speed dynamics.
* **Feature Extraction & Downsampling:** A 1D Convolutional layer (`Conv1D`, 64 filters, kernel size 5) extracts local temporal stroke features, followed by `MaxPool1D(2)` to downsample the sequence length by half for efficient CTC alignment.
* **Bidirectional Sequence Modeling:** A 2-layer stacked `Bidirectional LSTM` (128 units each, 256 effective units per layer) captures forward and backward temporal context across multi-stroke mathematical expressions.
* **Alignment-Free CTC Decoding:** Trained with Connectionist Temporal Classification (`ctc_batch_cost`) to map unaligned stroke sequences directly to LaTeX token sequences without requiring frame-level alignments.

---

## 📊 Evaluation & Benchmark Results

The model was evaluated across 300 unseen validation samples from the CROHME validation split:

| Metric | Validation Score | Technical Analysis |
| :--- | :--- | :--- |
| **Token Precision** | **81.60%** | High accuracy in recognizing individual mathematical symbols and numbers |
| **Token Recall** | **47.82%** | Sensitivity across multi-token and nested mathematical sequences |
| **Token F1-Score** | **60.30%** | Strong harmonic mean across token-level predictions |
| **Validation Loss** | **24.75** | Best checkpoint convergence (Epoch 71) |
| **Exact-Match Accuracy** | **3.67%** | Strict sequence-level equality across entire mathematical formulas |

### 🔍 Technical Discussion: Token Precision vs. Exact Match
* **Sequence Length Sensitivity:** In mathematical expression recognition, formulas frequently contain 15 to 30+ tokens. Exact-match accuracy requires **every single token in the sequence to be predicted with 100% precision**. If a 25-token formula has 24 correct tokens and 1 minor misclassification, the Token Precision is **96%**, but the Exact Match score is **0%**.
* **1D Temporal vs. 2D Spatial Structure:** A 1D sequence architecture (Conv1D + BiLSTM + CTC) excels at sequential symbol transcription (achieving **81.6% Token Precision**). However, non-linear 2D mathematical structures (nested fractions, exponents, square roots, matrices) inherently benefit from 2D attention tree parsers (e.g., Tree-LSTM or WAP) to boost full-sequence exact matches.

---

## 📂 Dataset Details & Reproducibility

* **Dataset:** Competition on Recognition of Online Handwritten Mathematical Expressions (**CROHME**)
* **Samples:** 21,265 mathematical expressions
* **Vocabulary:** 518 unique mathematical tokens (digits, variables, Greek letters, operators, structural symbols)
* **Kaggle Source:** [suyashma/math-expression1 on Kaggle](https://www.kaggle.com/datasets/suyashma/math-expression1)

> **Note on Dataset File:** The preprocessed dataset `processed_crohme.pkl` is ~196 MB, exceeding standard GitHub file limits. If cloning fresh, download `processed_crohme.pkl` directly from the [Kaggle Dataset Link](https://www.kaggle.com/datasets/suyashma/math-expression1) and place it into the `dataset/` directory.

---

## 📁 Repository Structure

```
MathInkAI/
├── dataset/
│   └── README.md                         # Dataset schema & Kaggle download link
├── model/
│   ├── best_mathink_model.weights.h5     # Best validation weights (Epoch 71)
│   ├── vocabulary.pkl                   # 518 vocabulary tokens
│   ├── token_to_id.pkl                  # Token-to-ID mapping
│   ├── id_to_token.pkl                  # ID-to-Token mapping
│   ├── model_config.json                # Model hyperparameters
│   └── build_vocab.py                   # Vocabulary generation utility
├── mathink_ai.py                        # Modular inference API (MathInkRecognizer)
├── MathInk_AI_Pipeline.ipynb            # Complete end-to-end training & evaluation notebook
├── mathink_ai_pipeline.py               # Standalone pipeline training script
├── requirements.txt                     # Dependencies
└── README.md                            # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/SuyashMathur-bit/MathInkAI.git
cd MathInkAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Inference with Pre-trained Weights
```python
from mathink_ai import MathInkRecognizer

# Initialize recognizer with pre-trained weights
recognizer = MathInkRecognizer(
    weights_path="model/best_mathink_model.weights.h5",
    vocab_path="model/vocabulary.pkl"
)

# Predict LaTeX from online stroke coordinate sequence
# strokes: list of stroke arrays [[[x, y], ...], [[x, y], ...]]
latex_output = recognizer.predict_strokes(sample_strokes)
print(f"Predicted LaTeX: {latex_output}")
```

### 4. Run Notebook Pipeline
Open `MathInk_AI_Pipeline.ipynb` in Jupyter Notebook, Google Colab, or Kaggle to explore the full data preprocessing, Conv1D-BiLSTM training, and CTC evaluation pipeline.

---

## 🛠 Technology Stack

* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Neural Architecture:** Conv1D + Stacked Bidirectional LSTM + CTC Loss
* **Language & Math Processing:** Python 3, NumPy, LaTeX Regex Tokenizer
* **Environment:** Jupyter Notebook / Google Colab / Kaggle GPU

---

## 👤 Author

* **Suyash Mathur** - [SuyashMathur-bit](https://github.com/SuyashMathur-bit)
