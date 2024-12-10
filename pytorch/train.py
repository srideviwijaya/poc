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

configure_logging("training.log")
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
# len_dl = len(dataloader)
# print(len_dl)

# sys.exit(0)

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
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Start training
epochs = 5
# print(vocab_size)
# print("Training started.")
logging.info("Training started")

train_start_time = time.time()
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to("cuda"), tgt.to("cuda")
        optimizer.zero_grad()

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        logging.info(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

    avg_loss = total_loss / len(dataloader)
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

