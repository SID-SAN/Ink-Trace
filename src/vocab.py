import pandas as pd
import json

def build_vocab(parquet_file):
    df = pd.read_parquet(parquet_file)
    # Collect all unique characters
    chars = set(''.join(df['text'].astype(str)))
    sorted_chars = sorted(list(chars))
    
    # Create mapping
    char2idx = {char: i + 1 for i, char in enumerate(sorted_chars)}
    char2idx['<PAD>'] = 0
    
    with open('char_map.json', 'w') as f:
        json.dump(char2idx, f)
    print(f"Vocabulary size: {len(char2idx)}")

if __name__ == "__main__":
    build_vocab('data/train.parquet')