import torch
import json
from torch.utils.data import DataLoader
from src.data_loader import InkTraceDataset, collate_fn
from src.model import InkTraceModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("char_map.json", "r") as f:
    vocab = json.load(f)

idx2char = {v: k for k, v in vocab.items()}

# Load Model
model = InkTraceModel(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load("checkpoints/inktrace_epoch_50.pth"))
model.eval()

# Dataset
dataset = InkTraceDataset(
    "data/test.parquet",
    "char_map.json"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# Decoder
def decode_prediction(pred_indices):
    decoded = []
    last_idx = -1

    for idx in pred_indices:
        idx = idx.item()

        if idx != 0 and idx != last_idx:
            decoded.append(idx)

        last_idx = idx

    return "".join(
        idx2char.get(i, "?")
        for i in decoded
    ).strip()


def decode_label(label_tensor):
    chars = []

    for idx in label_tensor:
        idx = idx.item()

        if idx != 0:
            chars.append(idx2char[idx])

    return "".join(chars)


# Levenshtein Distance
def edit_distance(s1, s2):
    m = len(s1)
    n = len(s2)

    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i

    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):

            cost = 0 if s1[i-1] == s2[j-1] else 1

            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[m][n]

# Evaluation
total_words = 0
correct_words = 0

total_chars = 0
total_char_errors = 0

examples_shown = 0

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=2)

        for pred, label in zip(predictions, labels):

            pred_text = decode_prediction(pred)
            gt_text = decode_label(label)

            total_words += 1

            if pred_text == gt_text:
                correct_words += 1

            dist = edit_distance(pred_text, gt_text)

            total_char_errors += dist
            total_chars += len(gt_text)

            if examples_shown < 20:

                print(f"GT   : {gt_text}")
                print(f"PRED : {pred_text}")
                print("-" * 50)

                examples_shown += 1

# Metrics
word_accuracy = 100 * correct_words / total_words

cer = (
    total_char_errors / total_chars
    if total_chars > 0
    else 0
)

print(f"Samples: {total_words}")
print(f"Word Accuracy: {word_accuracy:.2f}%")
print(f"Character Error Rate (CER): {cer:.4f}")