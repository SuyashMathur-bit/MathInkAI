import os
import sys
import json
import time
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from inference.preprocessing import strokes_to_sequence, tokenize_latex
from inference.decoding import greedy_ctc_decode, ids_to_tokens
from inference.model_loader import get_inference_model, load_vocab_assets, find_best_weights_path, reload_model_weights

DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed_crohme.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "training", "evaluation_report.json")

def edit_distance(seq1, seq2):
    """
    Exact Levenshtein distance implementation from project notebook Cell 47.
    """
    m = len(seq1)
    n = len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]

def evaluate_model(num_samples=300, weights_path=None):
    """
    Comprehensive validation evaluation preserving project methodology:
    - 300 validation samples
    - Token-level accuracy
    - Exact sequence accuracy
    - Normalized edit distance
    - Sequence similarity
    - Comparison with earlier benchmark milestones (Epoch 10, Epoch 20)
    """
    print("=" * 65)
    print("        MATHINK.AI MODEL EVALUATION PIPELINE")
    print("=" * 65)

    vocab, token_to_id, id_to_token = load_vocab_assets()
    pad_id = token_to_id.get("<pad>", 0)
    unk_id = token_to_id.get("<unk>", 1)

    print(f"Loading evaluation dataset: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data, _ = pickle.load(f)

    # Replicate exact train_test_split from notebook (test_size=0.2, random_state=42)
    X = []
    y = []
    for sample in data:
        seq = strokes_to_sequence(sample["strokes"])
        encoded = [token_to_id.get(t, unk_id) for t in tokenize_latex(sample["label"])]
        X.append(seq)
        y.append(encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    N = min(num_samples, len(X_val))
    print(f"Total Validation Samples Available : {len(X_val)}")
    print(f"Samples Selected for Evaluation     : {N}")

    # Prepare model
    reload_model_weights(weights_path)
    model = get_inference_model()
    current_weights = find_best_weights_path()
    print(f"Evaluating Model Checkpoint         : {current_weights or 'Base/Initialized Architecture'}")

    # Pad validation samples
    eval_seqs = X_val[:N]
    eval_labels = y_val[:N]

    input_lens = [len(s) for s in eval_seqs]
    downsampled_lens = [max(1, l // 2) for l in input_lens]
    max_len = max(input_lens)

    padded_eval_x = pad_sequences(eval_seqs, maxlen=max_len, padding="post", dtype="float32")

    print(f"\nRunning batch inference on {N} validation sequences...")
    t0 = time.time()
    predictions = model.predict(padded_eval_x, batch_size=16, verbose=1)
    infer_time = time.time() - t0
    print(f"Inference complete in {infer_time:.2f}s ({infer_time / N * 1000:.1f} ms/sample)")

    decoded_sequences = []
    for i in range(N):
        decoded, _ = greedy_ctc_decode(
            prediction=predictions[i],
            input_length=downsampled_lens[i],
            token_to_id=token_to_id,
            id_to_token=id_to_token
        )
        decoded_sequences.append(decoded)

    # Metrics calculation
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_distance = 0

    for i in range(N):
        actual = ids_to_tokens(eval_labels[i], token_to_id, id_to_token)
        pred = decoded_sequences[i]

        comp_len = min(len(actual), len(pred))
        for j in range(comp_len):
            if actual[j] == pred[j]:
                correct_tokens += 1

        total_tokens += len(actual)

        if actual == pred:
            exact_matches += 1

        dist = edit_distance(actual, pred)
        total_distance += dist

    token_accuracy = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    sequence_accuracy = (exact_matches / N * 100) if N > 0 else 0.0
    norm_edit_distance = (total_distance / total_tokens) if total_tokens > 0 else 1.0
    seq_similarity = max(0.0, (1.0 - norm_edit_distance) * 100)

    # Prior notebook benchmarks for comparison
    prior_benchmarks = {
        "epoch_10": {
            "token_accuracy": 36.43,
            "exact_sequence_accuracy": 9.00,
            "samples_tested": 300
        },
        "epoch_20": {
            "token_accuracy": 59.98,
            "exact_sequence_accuracy": 37.67,
            "sequence_similarity": 83.67,
            "samples_tested": 300
        }
    }

    report = {
        "samples_tested": N,
        "weights_evaluated": current_weights,
        "correct_tokens": correct_tokens,
        "total_actual_tokens": total_tokens,
        "token_level_accuracy_pct": round(token_accuracy, 2),
        "exact_matches": exact_matches,
        "exact_sequence_accuracy_pct": round(sequence_accuracy, 2),
        "total_edit_distance": total_distance,
        "normalized_edit_distance": round(norm_edit_distance, 4),
        "sequence_similarity_pct": round(seq_similarity, 2),
        "inference_time_total_s": round(infer_time, 2),
        "prior_benchmarks": prior_benchmarks,
        "sample_predictions": []
    }

    # Collect first 10 sample predictions for inspection
    for i in range(min(10, N)):
        actual = ids_to_tokens(eval_labels[i], token_to_id, id_to_token)
        pred = decoded_sequences[i]
        report["sample_predictions"].append({
            "sample_index": i + 1,
            "actual_tokens": actual,
            "actual_latex": "".join(actual),
            "predicted_tokens": pred,
            "predicted_latex": "".join(pred),
            "match": actual == pred
        })

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print clean evaluation report
    print("\n" + "=" * 65)
    print("                 FINAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Samples Tested            : {N}")
    print(f"Correct Tokens            : {correct_tokens}")
    print(f"Total Actual Tokens       : {total_tokens}")
    print(f"Token-Level Accuracy      : {token_accuracy:.2f}%")
    print(f"Exact Sequence Matches    : {exact_matches}")
    print(f"Exact Sequence Accuracy   : {sequence_accuracy:.2f}%")
    print(f"Total Edit Distance       : {total_distance}")
    print(f"Normalized Edit Distance  : {norm_edit_distance:.4f}")
    print(f"Sequence Similarity       : {seq_similarity:.2f}%")
    print("=" * 65)
    print("\nBenchmark Progression:")
    print("-----------------------------------------------------------------")
    print("Checkpoint      Token Acc      Exact Seq Acc    Seq Similarity")
    print("-----------------------------------------------------------------")
    print(f"Epoch 10        36.43%         9.00%            N/A")
    print(f"Epoch 20        59.98%         37.67%           83.67%")
    print(f"Current Model   {token_accuracy:.2f}%         {sequence_accuracy:.2f}%           {seq_similarity:.2f}%")
    print("-----------------------------------------------------------------")
    print(f"\nDetailed report saved to: {REPORT_PATH}")

    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300, help="Number of validation samples (default: 300)")
    parser.add_argument("--weights", type=str, default=None, help="Weights path to evaluate")
    args = parser.parse_args()
    evaluate_model(num_samples=args.samples, weights_path=args.weights)
