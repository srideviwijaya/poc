from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from torch.cuda.amp import autocast, GradScaler
import torch
import sys
import logging
import time
import math

def configure_logging(log_file):
    logging.basicConfig(
        # filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler()         # Log to the terminal
        ]
        )
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout

configure_logging("training.log")
logging.info(f"Check if cuda is available: {torch.cuda.is_available()}")
# Load dataset
# data_dir = "../bookcorpus"
# dataset = load_from_disk(data_dir)
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

logging.info("Dataset loaded.")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_data, batched=True, remove_columns=["text"])

# Load GPT model
config = GPT2Config(
    vocab_size=50257,       # Default vocabulary size for GPT-2
    n_embd=256,             # Hidden size (embedding size)
    n_layer=3,             # Number of layers
    n_head=4,              # Number of attention heads
    n_positions=1024        # Maximum sequence length
)

model = GPT2LMHeadModel(config).from_pretrained("gpt2").to("cuda")

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

# Training loop
train_dataloader = DataLoader(tokenized_datasets, batch_size=32, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

logging.info("Training started.")
# print(len(train_dataloader))

scaler = GradScaler()
train_start_time = time.time()
for epoch in range(3):  # 3 epochs
    model.train()
    start_time = time.time()
    total_loss = 0
    for batch in train_dataloader:
        inputs = batch['input_ids'].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

    avg_loss = total_loss / len(train_dataloader)
    perplexity = math.exp(avg_loss)
    # print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    end_time = time.time()
    # print(f"Per epoch time: {end_time - start_time} seconds")
    logging.info(f"Per epoch time: {end_time - start_time} seconds")

train_end_time = time.time()
logging.info(f"Total training time: {train_end_time - train_start_time} seconds")

save_directory = "model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

logging.info(f"Model saved.")

# Inferencing
def generate_text(prompt, model, tokenizer, max_length=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage of inferencing
prompt = "Once upon a time"
generated_text = generate_text(prompt, model, tokenizer)
logging.info(f"Generated text: {generated_text}")