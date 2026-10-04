import os
import sys

# Ensure UTF-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import glob
import re
import json
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed_crohme.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")
STATE_FILE = os.path.join(CHECKPOINTS_DIR, "checkpoint_state.json")

def load_vocabulary():
    vocab_path = os.path.join(MODEL_DIR, "vocabulary.pkl")
    t2i_path = os.path.join(MODEL_DIR, "token_to_id.pkl")
    i2t_path = os.path.join(MODEL_DIR, "id_to_token.pkl")

    if not os.path.exists(vocab_path):
        from model.build_vocab import build_vocabulary
        build_vocabulary()

    with open(vocab_path, "rb") as f:
        vocabulary = pickle.load(f)
    with open(t2i_path, "rb") as f:
        token_to_id = pickle.load(f)
    with open(i2t_path, "rb") as f:
        id_to_token = pickle.load(f)

    return vocabulary, token_to_id, id_to_token

def tokenize_latex(expression):
    expression = expression.replace("$", "")
    return re.findall(r'\\[a-zA-Z]+|[0-9]+|[a-zA-Z]|[^\s]', expression)

def strokes_to_sequence(strokes):
    sequence = []
    for stroke in strokes:
        for point_index, (x, y) in enumerate(stroke):
            pen = 1.0 if point_index == 0 else 0.0
            sequence.append([float(x), float(y), pen])
    return np.array(sequence, dtype=np.float32)

def prepare_dataset(data_path=DATA_PATH, max_samples=None):
    print(f"[MathInk.ai Data] Loading dataset from: {data_path}")
    with open(data_path, "rb") as f:
        data, _ = pickle.load(f)

    vocabulary, token_to_id, _ = load_vocabulary()
    unk_id = token_to_id.get("<unk>", 1)

    # Filter candidates with standard expression lengths (<= 450 points)
    candidates = []
    for sample in data:
        pts = sum(len(s) for s in sample["strokes"])
        if pts <= 450 and len(sample["strokes"]) <= 18:
            lbl = sample["label"].replace("$", "")
            if any(c in lbl for c in ['x', 'y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=']):
                candidates.append(sample)

    print(f"[MathInk.ai Data] Found {len(candidates)} high-quality algebraic/arithmetic expressions.")

    if max_samples and max_samples < len(candidates):
        print(f"[MathInk.ai Data] Subsampling {max_samples} targeted samples for training...")
        np.random.seed(42)
        indices = np.random.choice(len(candidates), size=max_samples, replace=False)
        data_selected = [candidates[i] for i in indices]
    else:
        data_selected = candidates[:max_samples] if max_samples else candidates

    X = []
    y = []
    for sample in data_selected:
        seq = strokes_to_sequence(sample["strokes"])
        if len(seq) > 400:
            seq = seq[:400]
        tokens = tokenize_latex(sample["label"])
        encoded = [token_to_id.get(t, unk_id) for t in tokens]
        
        # CTC constraint check: required tokens cannot exceed downsampled sequence length
        req_len = len(encoded)
        for k in range(1, len(encoded)):
            if encoded[k] == encoded[k - 1]:
                req_len += 1
        if req_len > (len(seq) // 2) or len(encoded) == 0:
            continue

        X.append(seq)
        y.append(encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    max_length = min(400, max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_val)))
    max_label_length = max(max(len(lbl) for lbl in y_train), max(len(lbl) for lbl in y_val))

    print(f"[MathInk.ai Data] Training: {len(X_train)} samples | Validation: {len(X_val)} samples")
    print(f"[MathInk.ai Data] Max sequence length: {max_length} | Max label length: {max_label_length}")

    return X_train, X_val, y_train, y_val, max_length, max_label_length, vocabulary, token_to_id

def build_ctc_model(max_length, max_label_length, vocab_size):
    """
    Exact model architecture from the project notebook:
    Conv1D(64, 5) -> MaxPool1D(2) -> BiLSTM(128) -> BiLSTM(128) -> Dense(vocab_size)
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

def get_checkpoint_state():
    """
    Inspect checkpoint state file and disk to find highest completed epoch.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    state = {
        "completed_epoch": 0,
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_checkpoint_path": None,
        "history": []
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
                state.update(saved_state)
        except Exception as e:
            print(f"[MathInk.ai Checkpoint] Notice: Error reading {STATE_FILE}: {e}")

    # Cross-verify with actual files on disk
    ckpt_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "mathink_checkpoint_epoch_*.weights.h5"))
    detected_epochs = []
    for f in ckpt_files:
        match = re.search(r"epoch_(\d+)\.weights\.h5", f)
        if match:
            detected_epochs.append(int(match.group(1)))

    if detected_epochs:
        max_detected = max(detected_epochs)
        if max_detected > state["completed_epoch"]:
            state["completed_epoch"] = max_detected

    return state

def save_checkpoint_state(state):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def train_model(total_epochs=50, batch_size=16, max_samples=None, force_restart=False):
    """
    Resumable training pipeline:
    1. Checks latest completed epoch.
    2. Resumes from completed_epoch + 1 up to total_epochs (50).
    3. Saves mathink_checkpoint_epoch_{epoch:02d}.weights.h5 every epoch.
    4. Updates best_mathink_model whenever validation loss improves.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    state = get_checkpoint_state()

    if force_restart:
        print("[MathInk.ai Train] Force restart specified. Resetting checkpoint state.")
        state = {
            "completed_epoch": 0,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
            "best_checkpoint_path": None,
            "history": []
        }
        save_checkpoint_state(state)

    start_epoch = state["completed_epoch"]
    print("=" * 65)
    print("           MATHINK.AI RESUMABLE TRAINING PIPELINE")
    print("=" * 65)
    print(f"Target Total Epochs : {total_epochs}")
    print(f"Completed Epochs    : {start_epoch}")

    if start_epoch >= total_epochs:
        print(f"[MathInk.ai Train] Target of {total_epochs} epochs already reached! Skipping retraining.")
        print(f"Best Epoch: {state['best_epoch']} with Val Loss: {state['best_val_loss']:.4f}")
        return state

    print(f"Resuming Training   : Epoch {start_epoch + 1} -> Epoch {total_epochs}")
    print("=" * 65)

    # 1. Load Data
    X_train, X_val, y_train, y_val, max_length, max_label_length, vocabulary, token_to_id = prepare_dataset(
        max_samples=max_samples
    )

    pad_id = token_to_id["<pad>"]

    # Pad sequences
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding="post", dtype="float32")
    X_val_padded = pad_sequences(X_val, maxlen=max_length, padding="post", dtype="float32")

    y_train_padded = np.full((len(y_train), max_label_length), pad_id, dtype=np.int32)
    for i, lbl in enumerate(y_train):
        y_train_padded[i, :len(lbl)] = lbl

    y_val_padded = np.full((len(y_val), max_label_length), pad_id, dtype=np.int32)
    for i, lbl in enumerate(y_val):
        y_val_padded[i, :len(lbl)] = lbl

    # Input length is downsampled by factor of 2 by MaxPooling1D
    in_lens_train = (np.array([len(s) for s in X_train], dtype=np.int32) // 2).reshape(-1, 1)
    in_lens_val = (np.array([len(s) for s in X_val], dtype=np.int32) // 2).reshape(-1, 1)

    lbl_lens_train = np.array([len(l) for l in y_train], dtype=np.int32).reshape(-1, 1)
    lbl_lens_val = np.array([len(l) for l in y_val], dtype=np.int32).reshape(-1, 1)

    train_inputs = {
        "stroke_input": X_train_padded,
        "labels": y_train_padded,
        "input_length": in_lens_train,
        "label_length": lbl_lens_train
    }
    val_inputs = {
        "stroke_input": X_val_padded,
        "labels": y_val_padded,
        "input_length": in_lens_val,
        "label_length": lbl_lens_val
    }
    dummy_train = np.zeros((len(X_train), 1), dtype=np.float32)
    dummy_val = np.zeros((len(X_val), 1), dtype=np.float32)

    # 2. Build Models
    ctc_model, inference_model = build_ctc_model(max_length, max_label_length, len(vocabulary))

    # 3. Load Checkpoint if resuming
    if start_epoch > 0:
        resume_weights = os.path.join(CHECKPOINTS_DIR, f"mathink_checkpoint_epoch_{start_epoch:02d}.weights.h5")
        if os.path.exists(resume_weights):
            print(f"[MathInk.ai Resume] Loading weights from: {resume_weights}")
            ctc_model.load_weights(resume_weights)
        else:
            print(f"[MathInk.ai Resume] Warning: Checkpoint file {resume_weights} not found. Searching latest...")
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "mathink_checkpoint_epoch_*.weights.h5")))
            if ckpts:
                print(f"[MathInk.ai Resume] Loading fallback latest: {ckpts[-1]}")
                ctc_model.load_weights(ckpts[-1])
            else:
                print("[MathInk.ai Resume] No weight files found. Starting training from scratch.")
                start_epoch = 0

    best_val_loss = state.get("best_val_loss", float("inf"))
    best_epoch = state.get("best_epoch", 0)

    # 4. Training Loop: Epoch by Epoch for guaranteed resumability
    for current_epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()
        print(f"\n>>> Running Epoch {current_epoch}/{total_epochs}...")

        # Learning rate schedule: drop slightly for fine-tuning
        if current_epoch > 70:
            current_lr = 0.00025
        elif current_epoch > 45:
            current_lr = 0.0005
        else:
            current_lr = 0.001
        ctc_model.optimizer.learning_rate = current_lr

        history = ctc_model.fit(
            train_inputs,
            dummy_train,
            validation_data=(val_inputs, dummy_val),
            batch_size=batch_size,
            epochs=current_epoch,
            initial_epoch=current_epoch - 1,
            verbose=1
        )

        epoch_time = time.time() - t0
        train_loss = float(history.history["loss"][0])
        val_loss = float(history.history["val_loss"][0])

        # Checkpoint file path
        ckpt_filename = f"mathink_checkpoint_epoch_{current_epoch:02d}.weights.h5"
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_filename)
        ctc_model.save_weights(ckpt_path)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = current_epoch
            # Save best model to both checkpoints and model/ directory
            best_ckpt = os.path.join(CHECKPOINTS_DIR, "best_mathink_model.weights.h5")
            best_model_dir = os.path.join(MODEL_DIR, "best_mathink_model.weights.h5")
            ctc_model.save_weights(best_ckpt)
            ctc_model.save_weights(best_model_dir)

        # Update state
        state["completed_epoch"] = current_epoch
        state["best_epoch"] = best_epoch
        state["best_val_loss"] = best_val_loss
        state["best_checkpoint_path"] = os.path.join(MODEL_DIR, "best_mathink_model.weights.h5")
        state["history"].append({
            "epoch": current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "is_best": is_best,
            "time_seconds": round(epoch_time, 2)
        })

        save_checkpoint_state(state)

        best_star = " * [NEW BEST MODEL SAVED]" if is_best else ""
        print(f"[Epoch {current_epoch:02d}/{total_epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s{best_star}")
        print(f"[Checkpoint Saved] -> {ckpt_path}")

    # 5. Save Final Model (Epoch 50)
    final_model_path = os.path.join(MODEL_DIR, "final_mathink_model.weights.h5")
    ctc_model.save_weights(final_model_path)
    print(f"\n[MathInk.ai] Training complete! Total 50 epochs finished.")
    print(f"[MathInk.ai] Final model saved: {final_model_path}")
    print(f"[MathInk.ai] Best validation model: Epoch {best_epoch} (val_loss: {best_val_loss:.4f})")

    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MathInk.ai Resumable CTC Training")
    parser.add_argument("--epochs", type=int, default=50, help="Target total epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subset size for fast training")
    parser.add_argument("--force-restart", action="store_true", help="Restart from epoch 1")
    args = parser.parse_args()

    train_model(
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        force_restart=args.force_restart
    )
