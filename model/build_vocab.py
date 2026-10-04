import os
import re
import pickle
import json
from collections import Counter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed_crohme.pkl")
MODEL_DIR = os.path.dirname(__file__)

def tokenize_latex(expression):
    expression = expression.replace("$", "")
    tokens = re.findall(
        r'\\[a-zA-Z]+|[0-9]+|[a-zA-Z]|[^\s]',
        expression
    )
    return tokens

def build_vocabulary():
    print(f"Loading dataset from: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data, skipped = pickle.load(f)

    print(f"Loaded {len(data)} samples (skipped: {len(skipped)})")

    token_counter = Counter()
    for sample in data:
        tokens = tokenize_latex(sample["label"])
        token_counter.update(tokens)

    special_tokens = ["<pad>", "<unk>"]
    unique_tokens = sorted(token_counter.keys())
    vocabulary = special_tokens + unique_tokens + ["<blank>"]

    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"PAD ID: {token_to_id['<pad>']}")
    print(f"UNK ID: {token_to_id['<unk>']}")
    print(f"BLANK ID: {token_to_id['<blank>']}")

    vocab_path = os.path.join(MODEL_DIR, "vocabulary.pkl")
    t2i_path = os.path.join(MODEL_DIR, "token_to_id.pkl")
    i2t_path = os.path.join(MODEL_DIR, "id_to_token.pkl")
    config_path = os.path.join(MODEL_DIR, "model_config.json")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocabulary, f)
    with open(t2i_path, "wb") as f:
        pickle.dump(token_to_id, f)
    with open(i2t_path, "wb") as f:
        pickle.dump(id_to_token, f)

    config = {
        "vocab_size": len(vocabulary),
        "pad_id": token_to_id["<pad>"],
        "unk_id": token_to_id["<unk>"],
        "blank_id": token_to_id["<blank>"],
        "max_length": 3150,
        "max_label_length": 129,
        "conv1d_filters": 64,
        "conv1d_kernel": 5,
        "pool_size": 2,
        "lstm_units": 128
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Vocabulary and config assets created successfully.")

if __name__ == "__main__":
    build_vocabulary()
