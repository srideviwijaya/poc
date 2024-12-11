from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
# from transformers import GPT2Tokenizer
from collections import Counter
import sys
import logging

from model import *
from dataset import *
from functions import *

import time

configure_logging("infer.log")
logging.info(f"Check if cuda is available: {torch.cuda.is_available()}")

# sys.exit(0)

# Load dataset
# data_dir = "./wikitext-103-v1-train"
# dataset = load_from_disk(data_dir)
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
text_data = " ".join(dataset["text"])

logging.info("Dataset loaded")

# Tokenize the text at the word level
tokenizer = get_tokenizer("basic_english")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
tokens = tokenizer(text_data)

# print("Tokenization completed.")
logging.info("Tokenization completed")

# Create vocabulary
vocab = {word: idx for idx, (word, _) in enumerate(Counter(tokens).items(), start=2)}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
vocab_size = len(vocab)

# Encode tokens as integers
encoded_text = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

# Dataset creation
sequence_length = 32
dataset = WikiTextDataset(encoded_text, sequence_length)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Model definition
src_vocab_size = vocab_size
tgt_vocab_size = vocab_size
d_model = 256
num_heads = 4
num_layers = 3
d_ff = 1024
max_seq_length = 32
dropout = 0.1

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
model = model.to("cuda")

# Inferencing
logging.info("Starting inference")

model_save_path = "transformer_model.pth"
# Load the model state
def load_model(path, model):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Reload the trained model
model = load_model(model_save_path, model)
model = model.to("cuda")

# Example input sequence
sample_text = "The quick brown fox jumps over the lazy dog"
sample_tokens = tokenizer(sample_text)
sample_encoded = [vocab.get(word, vocab["<UNK>"]) for word in sample_tokens]
sample_input = torch.tensor(sample_encoded).unsqueeze(0).to("cuda")  # Add batch dimension

# Perform inference
with torch.no_grad():
    output_logits = model(sample_input, sample_input)
    output_probs = F.softmax(output_logits, dim=-1)

# Decode the output
output_indices = torch.argmax(output_probs, dim=-1).squeeze().tolist()
decoded_output = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in output_indices if idx in vocab.values()]

logging.info(f"Input: {sample_text}")
logging.info(f"Output: {' '.join(decoded_output)}")

print(f"Input: {sample_text}")
print(f"Output: {' '.join(decoded_output)}")
