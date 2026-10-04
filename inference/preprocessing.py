import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_latex(expression):
    """
    Tokenize LaTeX expression using the exact regex pattern from the project.
    """
    expression = expression.replace("$", "")
    tokens = re.findall(
        r'\\[a-zA-Z]+|[0-9]+|[a-zA-Z]|[^\s]',
        expression
    )
    return tokens

def encode_label(label, token_to_id):
    """
    Encode LaTeX string into token ID sequence.
    """
    tokens = tokenize_latex(label)
    encoded = []
    unk_id = token_to_id.get("<unk>", 1)
    for token in tokens:
        encoded.append(token_to_id.get(token, unk_id))
    return encoded

def strokes_to_sequence(strokes):
    """
    Convert raw stroke list into numpy array sequence of [x, y, pen].
    pen = 1 for point_index == 0 (start of stroke), 0 otherwise.
    """
    sequence = []
    for stroke in strokes:
        for point_index, pt in enumerate(stroke):
            # Support both [x, y] and [x, y, ...] inputs
            x = float(pt[0])
            y = float(pt[1])
            pen = 1.0 if point_index == 0 else 0.0
            sequence.append([x, y, pen])

    if not sequence:
        return np.empty((0, 3), dtype=np.float32)

    return np.array(sequence, dtype=np.float32)

def normalize_canvas_strokes(strokes):
    """
    Normalize user drawn canvas strokes to match the CROHME training data:
    1. Filters out accidental stray noise clicks (single isolated points).
    2. Independent min-max normalization to [0.0, 1.0] for both X and Y.
    """
    if not strokes:
        return []

    # 1. Filter out accidental single-point noise dots if other real strokes exist
    valid_strokes = []
    for stroke in strokes:
        if len(stroke) >= 2:
            valid_strokes.append(stroke)
        elif len(stroke) == 1 and len(strokes) <= 3:
            # Keep if user only wrote 1-3 tiny marks (like dots in i or j)
            valid_strokes.append(stroke)

    if not valid_strokes:
        valid_strokes = strokes

    # 2. Compute bounding box across all points
    all_x = []
    all_y = []
    for stroke in valid_strokes:
        for pt in stroke:
            all_x.append(float(pt[0]))
            all_y.append(float(pt[1]))

    if not all_x:
        return []

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width = max_x - min_x
    height = max_y - min_y

    # Prevent division by zero
    scale_x = width if width > 1e-4 else 1.0
    scale_y = height if height > 1e-4 else 1.0

    # 3. Independent Min-Max Normalization: x in [0, 1], y in [0, 1]
    normalized_strokes = []
    for stroke in valid_strokes:
        norm_stroke = []
        for pt in stroke:
            nx = (float(pt[0]) - min_x) / scale_x
            ny = (float(pt[1]) - min_y) / scale_y
            # Clamp to [0.0, 1.0]
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            norm_stroke.append([nx, ny])
        normalized_strokes.append(norm_stroke)

    return normalized_strokes

def prepare_stroke_input(strokes, max_length=None):
    """
    Complete pipeline: normalize strokes -> convert to [x, y, pen] -> reshape for model.
    Returns:
        sequence_array: shape (1, seq_len, 3)
        input_length: int (seq_len // 2) for CTC downsampled length
    """
    norm_strokes = normalize_canvas_strokes(strokes)
    seq = strokes_to_sequence(norm_strokes)

    if len(seq) == 0:
        return None, 0

    seq_len = len(seq)
    downsampled_len = max(1, seq_len // 2)

    # If max_length is specified, pad with post-zeros
    if max_length is not None and seq_len < max_length:
        padded = pad_sequences([seq], maxlen=max_length, padding="post", dtype="float32")
        return padded, downsampled_len

    # Shape: (1, seq_len, 3)
    batch_input = np.expand_dims(seq, axis=0)
    return batch_input, downsampled_len
