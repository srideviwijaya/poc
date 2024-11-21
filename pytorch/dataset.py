import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class WikiTextDataset(Dataset):
    def __init__(self, encoded_text, seq_len):
        self.encoded_text = encoded_text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.encoded_text) - 2 * self.seq_len

    def __getitem__(self, idx):
        source = self.encoded_text[idx: idx + self.seq_len]
        target = self.encoded_text[idx + self.seq_len: idx + 2 * self.seq_len]
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def build_vocab_from_parquet(file_path, tokenizer):
    # Load the Parquet dataset
    df = pd.read_parquet(file_path)

    # Concatenate all text into a single string
    text = " ".join(df["text"].tolist())

    # Tokenize and count occurrences
    tokens = tokenizer(text)
    counter = Counter(tokens)

    # Create a vocabulary with indices
    vocab = {token: idx for idx, (token, _) in enumerate(counter.items(), start=1)}
    vocab["<unk>"] = 0  # Add unknown token
    return vocab