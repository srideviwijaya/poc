from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch
import math

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from collections import Counter
import sys
import logging

from model import *
from dataset import *
from functions import *

import time

configure_logging("training.log")
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# sys.exit(0)

# Load and preprocess dataset
logging.info("Loading dataset...")
data_dir = "./wikitext-103-v1-train"
dataset = load_from_disk(data_dir)
# dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
text_data = " ".join(dataset["text"])

logging.info("Dataset loaded.")

# Tokenize and build vocabulary
sp = spm.SentencePieceProcessor()
sp.load("sentencepiece.bpe.model")

# Get vocabulary
vocab_size = sp.get_piece_size()
vocab = [(sp.id_to_piece(i), i) for i in range(vocab_size)]
vocab_size = len(vocab)

logging.info("Tokenizing dataset...")
# tokenizer = get_tokenizer("basic_english")
encoded_text = sp.encode(text_data, out_type=int)

logging.info("Tokenization completed")

# Encode tokens as integers
# encoded_text = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

# Dataset creation
sequence_length = 32
dataset = WikiTextDataset(encoded_text, sequence_length)
batch_size = 32
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,  # Adjust based on CPU cores
    pin_memory=True
)

logging.info(f"DataLoader initialized with {len(train_dataloader)} batches.")
logging.info(f"Total training steps: {len(train_dataloader)*5} steps")

# Model definition
src_vocab_size = vocab_size
tgt_vocab_size = vocab_size
d_model = 256
num_heads = 4
num_layers = 3
d_ff = 1024
max_seq_length = 32
dropout = 0.1

model = Transformer(
    src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, sequence_length, dropout
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scaler = GradScaler()

# Training loop
epochs = 5
gradient_accumulation_steps = 4
logging.info("Starting training...")

train_start_time = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    for step, (src, tgt) in enumerate(train_dataloader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        with autocast(enabled=False):
            logits = model(src, tgt_input)
            loss = criterion(logits.view(-1, vocab_size), tgt_output.reshape(-1)) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

        # Log periodically
        if step % 50 == 0:
            logging.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    perplexity = math.exp(avg_loss)
    # print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    end_time = time.time()
    # print(f"Per epoch time: {end_time - start_time} seconds")
    logging.info(f"Per epoch time: {end_time - start_time} seconds")

train_end_time = time.time()
# print(f"Total training time: {train_end_time - train_start_time} seconds")
logging.info(f"Total training time: {train_end_time - train_start_time} seconds")

# Define the path to save the model
model_save_path = "transformer_model.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")
logging.info(f"Model saved to {model_save_path}")

# Inferencing
logging.info("Starting inference")

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
