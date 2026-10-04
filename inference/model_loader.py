import os
import glob
import time
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from .preprocessing import prepare_stroke_input
from .decoding import greedy_ctc_decode
from .latex import tokens_to_latex
from .solver import create_solution_output

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")

_MODEL_INSTANCE = None
_VOCABULARY = None
_TOKEN_TO_ID = None
_ID_TO_TOKEN = None
_LOADED_WEIGHTS_PATH = None

def load_vocab_assets():
    global _VOCABULARY, _TOKEN_TO_ID, _ID_TO_TOKEN
    if _VOCABULARY is None:
        vocab_path = os.path.join(MODEL_DIR, "vocabulary.pkl")
        t2i_path = os.path.join(MODEL_DIR, "token_to_id.pkl")
        i2t_path = os.path.join(MODEL_DIR, "id_to_token.pkl")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary assets missing in {MODEL_DIR}. Run model/build_vocab.py first.")

        with open(vocab_path, "rb") as f:
            _VOCABULARY = pickle.load(f)
        with open(t2i_path, "rb") as f:
            _TOKEN_TO_ID = pickle.load(f)
        with open(i2t_path, "rb") as f:
            _ID_TO_TOKEN = pickle.load(f)

    return _VOCABULARY, _TOKEN_TO_ID, _ID_TO_TOKEN

def build_inference_model(vocab_size):
    """
    Construct the EXACT CNN + BiLSTM model architecture from the notebook.
    Using shape=(None, 3) allows arbitrary input sequence lengths at inference time.
    """
    stroke_input = layers.Input(shape=(None, 3), name="stroke_input")
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(stroke_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    token_output = layers.Dense(vocab_size, activation="softmax", name="token_output")(x)

    model = Model(inputs=stroke_input, outputs=token_output, name="mathink_inference")
    return model

def find_best_weights_path():
    """
    Detect the best model checkpoint according to project priorities:
    1. model/best_mathink_model.weights.h5
    2. model/final_mathink_model.weights.h5
    3. Latest training/checkpoints/mathink_checkpoint_epoch_*.weights.h5
    """
    best_target = os.path.join(MODEL_DIR, "best_mathink_model.weights.h5")
    if os.path.exists(best_target):
        return best_target

    final_target = os.path.join(MODEL_DIR, "final_mathink_model.weights.h5")
    if os.path.exists(final_target):
        return final_target

    # Search in training/checkpoints
    ckpt_best = os.path.join(CHECKPOINTS_DIR, "best_mathink_model.weights.h5")
    if os.path.exists(ckpt_best):
        return ckpt_best

    all_ckpts = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "mathink_checkpoint_epoch_*.weights.h5")))
    if all_ckpts:
        return all_ckpts[-1]

    return None

def get_inference_model():
    """
    Singleton accessor for the loaded inference model.
    """
    global _MODEL_INSTANCE, _LOADED_WEIGHTS_PATH
    vocab, t2i, i2t = load_vocab_assets()

    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = build_inference_model(len(vocab))
        weights_path = find_best_weights_path()
        if weights_path and os.path.exists(weights_path):
            try:
                _MODEL_INSTANCE.load_weights(weights_path)
                _LOADED_WEIGHTS_PATH = weights_path
                print(f"[MathInk.ai] Loaded weights successfully from: {weights_path}")
            except Exception as e:
                print(f"[MathInk.ai] Warning: Could not load weights from {weights_path}: {e}")
        else:
            print("[MathInk.ai] No trained weights found yet; initialized base model architecture.")

    return _MODEL_INSTANCE

def reload_model_weights(weights_path=None):
    """
    Force reload of model weights after new checkpoints are saved.
    """
    global _MODEL_INSTANCE, _LOADED_WEIGHTS_PATH
    vocab, t2i, i2t = load_vocab_assets()
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = build_inference_model(len(vocab))

    if weights_path is None:
        weights_path = find_best_weights_path()

    if weights_path and os.path.exists(weights_path):
        _MODEL_INSTANCE.load_weights(weights_path)
        _LOADED_WEIGHTS_PATH = weights_path
        print(f"[MathInk.ai] Reloaded model weights from: {weights_path}")
        return True
    return False

def get_loaded_checkpoint_info():
    """
    Return currently loaded checkpoint metadata.
    """
    return {
        "weights_path": _LOADED_WEIGHTS_PATH,
        "filename": os.path.basename(_LOADED_WEIGHTS_PATH) if _LOADED_WEIGHTS_PATH else "Untrained Base Architecture",
        "has_weights": _LOADED_WEIGHTS_PATH is not None and os.path.exists(_LOADED_WEIGHTS_PATH)
    }

def run_pipeline_on_strokes(raw_strokes):
    """
    Full End-to-End Pipeline:
    raw_strokes -> normalize -> [x, y, pen] -> model -> CTC decode -> tokens_to_latex -> solve_latex.
    """
    t_start = time.time()
    vocab, t2i, i2t = load_vocab_assets()
    model = get_inference_model()

    if not raw_strokes or len(raw_strokes) == 0:
        return {
            "success": False,
            "error": "No handwriting detected. Please write an equation on the board.",
            "strokes_count": 0,
            "points_count": 0
        }

    # Count total points
    points_count = sum(len(s) for s in raw_strokes)
    if points_count < 2:
        return {
            "success": False,
            "error": "Too few handwriting points detected. Please write clearly on the board.",
            "strokes_count": len(raw_strokes),
            "points_count": points_count
        }

    # 1. Preprocess
    batch_input, downsampled_len = prepare_stroke_input(raw_strokes)
    if batch_input is None:
        return {
            "success": False,
            "error": "Could not process handwriting strokes.",
            "strokes_count": len(raw_strokes),
            "points_count": points_count
        }

    # 2. Predict with model
    t_model_start = time.time()
    prediction = model.predict(batch_input, verbose=0)[0]
    model_time_ms = round((time.time() - t_model_start) * 1000, 2)

    # 3. CTC Decoding
    decoded_tokens, decoded_ids = greedy_ctc_decode(
        prediction=prediction,
        input_length=downsampled_len,
        token_to_id=t2i,
        id_to_token=i2t
    )

    # 4. Tokens to LaTeX
    predicted_latex = tokens_to_latex(decoded_tokens)

    # 5. Solver
    solution_payload = create_solution_output(predicted_latex)

    total_time_ms = round((time.time() - t_start) * 1000, 2)
    ckpt_info = get_loaded_checkpoint_info()

    return {
        "success": True,
        "strokes_count": len(raw_strokes),
        "points_count": points_count,
        "input_shape": list(batch_input.shape),
        "downsampled_length": int(downsampled_len),
        "tokens": decoded_tokens,
        "token_ids": decoded_ids,
        "latex": predicted_latex,
        "expression_type": solution_payload.get("expression_type", "unknown"),
        "status": solution_payload.get("status", "unknown"),
        "solution": solution_payload.get("final_answer", ""),
        "steps": solution_payload.get("steps", []),
        "inference_time_ms": total_time_ms,
        "model_time_ms": model_time_ms,
        "checkpoint_info": ckpt_info
    }
