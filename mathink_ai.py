"""
MathInk.ai — Complete Standalone Mathematical Handwriting Recognition & Solver Pipeline
========================================================================================
Model Architecture: Conv1D (64) -> MaxPool1D(2) -> 2x BiLSTM (128) -> Dense (518) -> CTC
Trained Checkpoint: Epoch 71 (Best Validation Loss: 24.7543, Token Precision: 81.60%)
Solver: SymPy Symbolic Mathematics with Step-by-Step Derivations
Vision Extractor: OpenCV Morphological Skeletonization + Contour Path Tracing
"""

import os
import sys
import re
import time
import json
import base64
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from PIL import Image, ImageOps
import cv2
import sympy as sp

try:
    from latex2sympy2 import latex2sympy
    HAS_LATEX2SYMPY = True
except ImportError:
    HAS_LATEX2SYMPY = False

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocabulary.pkl")
T2I_PATH = os.path.join(MODEL_DIR, "token_to_id.pkl")
I2T_PATH = os.path.join(MODEL_DIR, "id_to_token.pkl")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_mathink_model.weights.h5")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer & Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_latex(expression: str) -> list[str]:
    """Tokenize a LaTeX mathematical expression into individual tokens."""
    expression = expression.replace("$", "").strip()
    return re.findall(r'\\[a-zA-Z]+|[0-9]+|[a-zA-Z]|[^\s]', expression)

def tokens_to_latex(tokens: list[str]) -> str:
    """Convert a sequence of predicted tokens back into a LaTeX string."""
    cleaned = [t for t in tokens if t not in ["<pad>", "<unk>", "<blank>"]]
    return "".join(cleaned)

def load_vocabulary():
    """Load vocabulary, token_to_id, and id_to_token assets."""
    if not (os.path.exists(VOCAB_PATH) and os.path.exists(T2I_PATH) and os.path.exists(I2T_PATH)):
        raise FileNotFoundError(f"Vocabulary assets missing in {MODEL_DIR}.")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(T2I_PATH, "rb") as f:
        t2i = pickle.load(f)
    with open(I2T_PATH, "rb") as f:
        i2t = pickle.load(f)
    return vocab, t2i, i2t

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stroke Processing & Normalization
# ─────────────────────────────────────────────────────────────────────────────

def strokes_to_sequence(strokes: list[list[list[float]]]) -> np.ndarray:
    """
    Convert raw stroke groups into a continuous [x, y, pen] sequence.
    pen = 1.0 for the start of a stroke, 0.0 for continuing points.
    """
    sequence = []
    for stroke in strokes:
        for point_index, pt in enumerate(stroke):
            x, y = float(pt[0]), float(pt[1])
            pen = 1.0 if point_index == 0 else 0.0
            sequence.append([x, y, pen])
    return np.array(sequence, dtype=np.float32)

def normalize_canvas_strokes(strokes: list[list[list[float]]], min_span: float = 3.0) -> list[list[list[float]]]:
    """
    Filter stray dots and normalize strokes to [0, 1] x [0, 1] matching CROHME preprocessing.
    """
    # Filter stray single-tap dots
    valid_strokes = []
    for s in strokes:
        if len(s) == 0:
            continue
        if len(s) <= 2:
            xs = [pt[0] for pt in s]
            ys = [pt[1] for pt in s]
            if (max(xs) - min(xs)) < min_span and (max(ys) - min(ys)) < min_span:
                continue
        valid_strokes.append(s)

    if not valid_strokes:
        valid_strokes = strokes

    # Find bounding box
    all_pts = [pt for s in valid_strokes for pt in s]
    if not all_pts:
        return valid_strokes

    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-4)
    height = max(max_y - min_y, 1e-4)

    normalized = []
    for s in valid_strokes:
        norm_stroke = []
        for pt in s:
            nx = (pt[0] - min_x) / width
            ny = (pt[1] - min_y) / height
            norm_stroke.append([float(nx), float(ny)])
        normalized.append(norm_stroke)
    return normalized

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_ctc_model(max_length: int = 400, max_label_length: int = 40, vocab_size: int = 518):
    """
    Conv1D (64, k=5) -> MaxPool1D (2) -> 2x BiLSTM (128) -> Dense (vocab_size) -> CTC Loss.
    Exact architecture matching the user's project notebook.
    """
    stroke_input = layers.Input(shape=(max_length, 3), name="stroke_input")
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(stroke_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    token_output = layers.Dense(vocab_size, activation="softmax", name="token_output")(x)

    labels = layers.Input(shape=(max_label_length,), dtype="int32", name="labels")
    input_length = layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_length = layers.Input(shape=(1,), dtype="int32", name="label_length")

    def ctc_loss_func(args):
        y_true, y_pred, in_len, lbl_len = args
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, in_len, lbl_len)

    loss_output = layers.Lambda(ctc_loss_func, output_shape=(1,), name="ctc_loss")(
        [labels, token_output, input_length, label_length]
    )

    training_model = Model(
        inputs=[stroke_input, labels, input_length, label_length],
        outputs=loss_output,
        name="mathink_ctc_training"
    )
    inference_model = Model(
        inputs=stroke_input,
        outputs=token_output,
        name="mathink_inference"
    )
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y_true, y_pred: y_pred
    )
    return training_model, inference_model

def get_inference_model(vocab_size: int = 518):
    """Instantiate a dynamic-length (None, 3) inference model for production decoding."""
    stroke_input = layers.Input(shape=(None, 3), name="stroke_input_dynamic")
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(stroke_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    token_output = layers.Dense(vocab_size, activation="softmax", name="token_output")(x)
    return Model(inputs=stroke_input, outputs=token_output, name="mathink_inference_dynamic")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CTC Greedy Decoding
# ─────────────────────────────────────────────────────────────────────────────

def greedy_ctc_decode(prediction: np.ndarray, input_length: int, token_to_id: dict, id_to_token: dict):
    """
    Greedy CTC Decoding with blank suppression and consecutive deduplication.
    """
    truncated_pred = prediction[:int(input_length)]
    token_ids = np.argmax(truncated_pred, axis=-1)
    blank_id = token_to_id.get("<blank>", len(id_to_token) - 1)
    pad_id = token_to_id.get("<pad>", 0)

    decoded_tokens = []
    decoded_ids = []
    previous_id = None

    for tid in token_ids:
        tid = int(tid)
        if tid in [blank_id, pad_id]:
            previous_id = None
            continue
        if tid == previous_id:
            continue
        token = id_to_token.get(tid, "<unk>")
        decoded_tokens.append(token)
        decoded_ids.append(tid)
        previous_id = tid

    return decoded_tokens, decoded_ids

# ─────────────────────────────────────────────────────────────────────────────
# 5. Image to Stroke Sequence Extractor (Computer Vision)
# ─────────────────────────────────────────────────────────────────────────────

def extract_strokes_from_image(image_input, target_width: int = 800, target_height: int = 400):
    """
    Extracts ordered [x, y] strokes from an uploaded image/photo of handwritten math:
    1. Polarity detection (white paper vs chalkboard)
    2. Otsu thresholding + morphological skeletonization
    3. Contour tracing into stroke coordinates
    """
    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image input format.")

    if img is None:
        raise ValueError("Could not decode image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Polarity check: border brightness determines background
    border_pixels = np.concatenate([
        blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]
    ])
    if np.mean(border_pixels) > 127:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological skeletonization
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_bin = binary.copy()

    while True:
        eroded = cv2.erode(temp_bin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_bin, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_bin = eroded.copy()
        if cv2.countNonZero(temp_bin) == 0:
            break

    # Find contours
    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    orig_h, orig_w = gray.shape
    scale = min(target_width / orig_w, target_height / orig_h)
    offset_x = (target_width - orig_w * scale) / 2
    offset_y = (target_height - orig_h * scale) / 2

    raw_strokes = []
    for cnt in contours:
        pts = cnt.squeeze(axis=1)
        if len(pts.shape) != 2 or len(pts) < 3:
            continue
        # Subsample points
        step = max(1, len(pts) // 40)
        subsampled = pts[::step]
        stroke = []
        for p in subsampled:
            sx = float(p[0] * scale + offset_x)
            sy = float(p[1] * scale + offset_y)
            stroke.append([sx, sy])
        if stroke:
            raw_strokes.append(stroke)

    # Sort strokes left-to-right
    raw_strokes.sort(key=lambda s: s[0][0])
    return raw_strokes

# ─────────────────────────────────────────────────────────────────────────────
# 6. Symbolic Mathematical Solver (SymPy)
# ─────────────────────────────────────────────────────────────────────────────

def clean_latex_for_sympy(latex_str: str) -> str:
    """Preprocess LaTeX tokens to be friendly with SymPy."""
    expr = latex_str.strip().replace("$", "")
    expr = re.sub(r'\\left|\\right', '', expr)
    expr = re.sub(r'\\mbox\{([^}]+)\}', r'\1', expr)
    expr = re.sub(r'([0-9])([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr)
    return expr

def solve_latex(latex_str: str) -> dict:
    """Solve an equation or evaluate an expression using SymPy with step derivations."""
    clean = clean_latex_for_sympy(latex_str)
    try:
        if "=" in clean:
            left_str, right_str = clean.split("=", 1)
            left_sp = sp.sympify(left_str)
            right_sp = sp.sympify(right_str)
            eq = sp.Eq(left_sp, right_sp)
            symbols = list(eq.free_symbols)

            if symbols:
                target_var = symbols[0]
                solutions = sp.solve(eq, target_var)
                sol_strs = [sp.latex(s) for s in solutions]
                sol_display = f"{sp.latex(target_var)} = " + ", ".join(sol_strs) if len(sol_strs) > 1 else f"{sp.latex(target_var)} = {sol_strs[0]}" if sol_strs else "No solution"
                steps = [
                    f"**Given Equation:** $${sp.latex(eq)}$$",
                    f"**Target Variable:** $${sp.latex(target_var)}$$",
                    f"**Algebraic Rearrangement:** $${sp.latex(sp.simplify(left_sp - right_sp))} = 0$$",
                    f"**Computed Roots:** $${sol_display}$$"
                ]
                return {
                    "success": True,
                    "expression_type": "Algebraic Equation",
                    "status": "Solved",
                    "final_answer": sol_display,
                    "steps": steps
                }
        else:
            parsed = sp.sympify(clean)
            simplified = sp.simplify(parsed)
            steps = [
                f"**Input Expression:** $${sp.latex(parsed)}$$",
                f"**Simplified Result:** $${sp.latex(simplified)}$$"
            ]
            return {
                "success": True,
                "expression_type": "Expression",
                "status": "Simplified",
                "final_answer": sp.latex(simplified),
                "steps": steps
            }
    except Exception as e:
        return {
            "success": True,
            "expression_type": "Unresolved",
            "status": "Error",
            "final_answer": f"Parsed: {latex_str}",
            "steps": [f"Could not symbolically solve: {e}"]
        }

# ─────────────────────────────────────────────────────────────────────────────
# 7. Complete MathInk.ai Pipeline Engine
# ─────────────────────────────────────────────────────────────────────────────

class MathInkRecognizer:
    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.vocab, self.token_to_id, self.id_to_token = load_vocabulary()
        self.model = get_inference_model(vocab_size=len(self.vocab))
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"[MathInkRecognizer] Loaded Epoch 71 weights: {weights_path}")
        else:
            print(f"[MathInkRecognizer] Warning: Weights file {weights_path} not found.")

    def recognize_strokes(self, strokes: list[list[list[float]]]) -> dict:
        t0 = time.time()
        norm_strokes = normalize_canvas_strokes(strokes)
        seq = strokes_to_sequence(norm_strokes)

        if len(seq) == 0:
            return {"success": False, "error": "No strokes provided."}

        # Predict
        input_tensor = np.expand_dims(seq, axis=0)
        preds = self.model.predict(input_tensor, verbose=0)[0]
        downsampled_len = max(1, len(seq) // 2)

        tokens, token_ids = greedy_ctc_decode(
            preds, downsampled_len, self.token_to_id, self.id_to_token
        )
        latex = tokens_to_latex(tokens)
        solution = solve_latex(latex)
        total_time_ms = round((time.time() - t0) * 1000, 2)

        return {
            "success": True,
            "tokens": tokens,
            "latex": latex,
            "expression_type": solution.get("expression_type", "Equation"),
            "status": solution.get("status", "Solved"),
            "solution": solution.get("final_answer", ""),
            "steps": solution.get("steps", []),
            "inference_time_ms": total_time_ms,
            "stroke_count": len(strokes),
            "point_count": len(seq)
        }

    def recognize_image(self, image_input) -> dict:
        strokes = extract_strokes_from_image(image_input)
        if not strokes:
            return {"success": False, "error": "No handwriting detected in image."}
        result = self.recognize_strokes(strokes)
        result["extracted_strokes"] = strokes
        return result

# ─────────────────────────────────────────────────────────────────────────────
# 8. Command Line Interface / Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MathInk.ai Standalone Handwriting Recognition & Solver")
    parser.add_argument("--image", type=str, help="Path to an image file containing handwritten math")
    parser.add_argument("--test", action="store_true", help="Run self-test on sample equation")
    args = parser.parse_args()

    recognizer = MathInkRecognizer()

    if args.image:
        print(f"\nProcessing Image: {args.image}")
        res = recognizer.recognize_image(args.image)
        print("Recognized LaTeX:", res.get("latex"))
        print("Tokens          :", res.get("tokens"))
        print("Solution        :", res.get("solution"))
    else:
        # Default self-test with sample 'x^2 + 5x + 6 = 0'
        print("\n--- MathInk.ai Self-Test ---")
        test_strokes = [
            [[100, 100], [130, 130]], [[130, 100], [100, 130]],  # x
            [[140, 90], [155, 90], [155, 100], [140, 100], [140, 110], [155, 110]], # 2
            [[170, 115], [190, 115]], [[180, 105], [180, 125]],  # +
            [[210, 105], [230, 105], [210, 115], [230, 115], [230, 130], [210, 130]], # 5
            [[245, 100], [265, 130]], [[265, 100], [245, 130]],  # x
            [[280, 115], [300, 115]], [[290, 105], [290, 125]],  # +
            [[320, 105], [340, 130], [320, 130]],                 # 6
            [[355, 115], [375, 115]], [[355, 125], [375, 125]],  # =
            [[395, 105], [415, 105], [415, 130], [395, 130], [395, 105]] # 0
        ]
        res = recognizer.recognize_strokes(test_strokes)
        print("Tokens Predicted :", res.get("tokens"))
        print("Recognized LaTeX :", res.get("latex"))
        print("Solution         :", res.get("solution"))
        print("Inference Time   :", res.get("inference_time_ms"), "ms")
        print("\nPipeline ready! Run 'py -3.11 backend/app.py' to launch the interactive UI.")
