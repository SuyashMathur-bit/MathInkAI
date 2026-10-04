import numpy as np

def greedy_ctc_decode(prediction, input_length, token_to_id, id_to_token):
    """
    Exact Greedy CTC Decoding implementation from the user notebook.
    Takes model output logits/probabilities, slices to input_length,
    finds argmax, removes blank tokens and duplicate consecutive tokens.
    """
    # Slice prediction to downsampled input length
    sliced_pred = prediction[:int(input_length)]

    token_ids = np.argmax(sliced_pred, axis=-1)
    blank_id = token_to_id.get("<blank>", len(id_to_token) - 1)

    decoded_tokens = []
    decoded_ids = []
    previous_token_id = None

    for token_id in token_ids:
        token_id = int(token_id)

        # Blank token resets consecutive duplicate suppression
        if token_id == blank_id:
            previous_token_id = None
            continue

        # Skip repeated tokens without an intervening blank
        if token_id == previous_token_id:
            continue

        token = id_to_token.get(token_id, "<unk>")
        decoded_tokens.append(token)
        decoded_ids.append(token_id)
        previous_token_id = token_id

    return decoded_tokens, decoded_ids

def ids_to_tokens(label_ids, token_to_id, id_to_token):
    """
    Convert ground truth label token IDs to token strings, skipping <pad>.
    """
    pad_id = token_to_id.get("<pad>", 0)
    tokens = []
    for token_id in label_ids:
        token_id = int(token_id)
        if token_id == pad_id:
            continue
        tokens.append(id_to_token.get(token_id, "<unk>"))
    return tokens
