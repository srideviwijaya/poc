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

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load dataset
logging.info("Loading dataset...")
data_dir = "./wikitext-103-v1-train"
dataset = load_from_disk(data_dir)
# dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

logging.info(f"Dataset loaded: {len(dataset)} samples")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_data, batched=True, remove_columns=["text"])

# Load GPT model
config = GPT2Config(
    vocab_size=50257,       # Default vocabulary size for GPT-2
    n_embd=256,             # Hidden size (embedding size)
    n_layer=3,             # Number of layers
    n_head=4,              # Number of attention heads
    n_positions=512        # Maximum sequence length
)

model = GPT2LMHeadModel(config).from_pretrained("gpt2").to("cuda")

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

# Training loop
# Dataloader
batch_size = 16
train_dataloader = DataLoader(
    tokenized_datasets,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8  # Adjust based on CPU cores
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
logging.info(f"Total training steps: {num_training_steps} steps")

lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

# Training loop
scaler = GradScaler()
logging.info("Starting training...")
epochs = 3
gradient_accumulation_steps = 2  # Simulate batch_size * 2
total_training_time = 0

train_start_time = time.time()
for epoch in range(3):  # 3 epochs
    model.train()
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast():  # Mixed precision training
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation
        
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

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